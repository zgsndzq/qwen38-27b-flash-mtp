#include "mem-card.h"
#include "mem-internal.h"

#include "ggml.h"
#include "gguf.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

using json = nlohmann::json;

namespace {

std::vector<llama_token> tokenize_body(const llama_vocab * vocab, const std::string & s) {
    std::vector<llama_token> buf(s.size() + 8);
    int n = llama_tokenize(vocab, s.c_str(), (int32_t) s.size(), buf.data(), (int32_t) buf.size(), false, true);
    if (n < 0) {
        buf.resize((size_t) -n);
        n = llama_tokenize(vocab, s.c_str(), (int32_t) s.size(), buf.data(), (int32_t) buf.size(), false, true);
    }
    if (n > 0) {
        buf.resize(n);
    } else {
        buf.clear();
    }
    return buf;
}

std::string index_text(const mem_card_entry & e) {
    if (e.tags.empty()) {
        return "title: " + e.title;
    }
    return "title: " + e.title + " | tags: " + e.tags;
}

// accumulated card state, old entries first
struct card_data {
    int32_t key_dim = 0;
    std::string name       = "memory";
    std::string doc_prefix = "title: memory | text: ";
    std::vector<int8_t>  keys;
    std::vector<float>   scales;
    std::vector<int32_t> tokens;
    std::vector<int32_t> offsets{0};
    std::vector<int32_t> title_tokens;
    std::vector<int32_t> title_offsets{0};
    std::vector<int32_t> tag_tokens;
    std::vector<int32_t> tag_offsets{0};
    std::vector<int32_t> pinned;
    json entries = json::array();

    int32_t count() const { return (int32_t) scales.size(); }
};

bool load_old(const std::string & path, card_data & cd, std::string & err) {
    mem_db db;
    if (!mem_load_db(path, db, err)) {
        return false;
    }
    if (!db.quantized()) {
        err = "append requires a v2 (int8 key) card: " + path;
        return false;
    }
    cd.key_dim    = db.key_dim;
    cd.name       = db.name;
    cd.doc_prefix = db.doc_prefix.empty() ? cd.doc_prefix : db.doc_prefix;
    cd.keys       = db.keys_i8;
    cd.scales     = db.key_scale;
    cd.tokens.assign(db.tokens.begin(), db.tokens.end());
    cd.offsets    = db.tok_offsets;
    cd.title_tokens.assign(db.title_tokens.begin(), db.title_tokens.end());
    cd.title_offsets = db.title_offsets.empty() ? std::vector<int32_t>(db.count + 1, 0) : db.title_offsets;
    cd.tag_tokens.assign(db.tag_tokens.begin(), db.tag_tokens.end());
    cd.tag_offsets = db.tag_offsets.empty() ? std::vector<int32_t>(db.count + 1, 0) : db.tag_offsets;
    cd.pinned = db.pinned.empty() ? std::vector<int32_t>(db.count, 0) : db.pinned;
    if (!db.entries_json.empty()) {
        try {
            cd.entries = json::parse(db.entries_json);
        } catch (const std::exception &) {
            cd.entries = json::array();
        }
    }
    if (!cd.entries.is_array()) {
        cd.entries = json::array();
    }
    while ((int32_t) cd.entries.size() < db.count) {
        cd.entries.push_back(json::object());
    }
    return true;
}

bool add_entries(card_data & cd, const std::vector<mem_card_entry> & entries,
                 const llama_vocab * vocab, const mem_card_writer::encode_fn & encode,
                 std::string & err) {
    for (const auto & e : entries) {
        if (e.title.empty() || e.body.empty()) {
            err = "card entry needs a title and a body";
            return false;
        }
        std::vector<float> key;
        if (!encode(index_text(e), key) || (int32_t) key.size() != cd.key_dim) {
            err = "encoder failed or key dim mismatch for entry: " + e.title;
            return false;
        }
        float amax = 0.0f;
        for (float v : key) {
            amax = std::max(amax, std::fabs(v));
        }
        if (amax < 1e-8f) {
            err = "zero embedding for entry: " + e.title;
            return false;
        }
        const float scale = amax / 127.0f;
        for (float v : key) {
            const int q = (int) std::lround(v / scale);
            cd.keys.push_back((int8_t) std::max(-127, std::min(127, q)));
        }
        cd.scales.push_back(scale);

        const auto body_ids = tokenize_body(vocab, e.body);
        if (body_ids.empty()) {
            err = "empty body tokens for entry: " + e.title;
            return false;
        }
        cd.tokens.insert(cd.tokens.end(), body_ids.begin(), body_ids.end());
        cd.offsets.push_back((int32_t) cd.tokens.size());

        const auto title_ids = tokenize_body(vocab, e.title);
        cd.title_tokens.insert(cd.title_tokens.end(), title_ids.begin(), title_ids.end());
        cd.title_offsets.push_back((int32_t) cd.title_tokens.size());

        const auto tag_ids = tokenize_body(vocab, e.tags);
        cd.tag_tokens.insert(cd.tag_tokens.end(), tag_ids.begin(), tag_ids.end());
        cd.tag_offsets.push_back((int32_t) cd.tag_tokens.size());

        cd.pinned.push_back(0);

        json meta = json::object();
        if (!e.meta_json.empty()) {
            try {
                meta = json::parse(e.meta_json);
            } catch (const std::exception &) {
                err = "bad meta_json for entry: " + e.title;
                return false;
            }
        }
        meta["title"]  = e.title;
        meta["tags"]   = e.tags;
        meta["tokens"] = (int32_t) body_ids.size();
        cd.entries.push_back(std::move(meta));
    }
    return true;
}

bool write_gguf(const std::string & path, const card_data & cd, std::string & err) {
    const int32_t n = cd.count();
    if (n <= 0 || cd.key_dim <= 0) {
        err = "cannot write an empty card";
        return false;
    }

    ggml_init_params gp = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 9,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(gp);
    gguf_context * g   = gguf_init_empty();
    if (ctx == nullptr || g == nullptr) {
        err = "gguf init failed";
        if (ctx) { ggml_free(ctx); }
        if (g)   { gguf_free(g); }
        return false;
    }

    gguf_set_val_u32(g, "mem.version", 2);
    gguf_set_val_u32(g, "mem.count", (uint32_t) n);
    gguf_set_val_u32(g, "mem.key_dim", (uint32_t) cd.key_dim);
    gguf_set_val_u32(g, "mem.hidden", 0);
    gguf_set_val_f32(g, "mem.threshold", 0.0f);
    gguf_set_val_str(g, "mem.name", cd.name.c_str());
    gguf_set_val_str(g, "mem.doc_prefix", cd.doc_prefix.c_str());
    gguf_set_val_str(g, "mem.rerank", "sim");
    const std::string entries_str = cd.entries.dump();
    gguf_set_val_str(g, "mem.entries", entries_str.c_str());

    const int32_t dummy = 0; // gguf needs a valid pointer even for a 1-element placeholder
    auto add = [&](const char * name, ggml_type type, int64_t ne0, int64_t ne1, const void * data) {
        ggml_tensor * t = ne1 > 0 ? ggml_new_tensor_2d(ctx, type, ne0, ne1)
                                  : ggml_new_tensor_1d(ctx, type, ne0);
        ggml_set_name(t, name);
        gguf_add_tensor(g, t);
        gguf_set_tensor_data(g, name, data);
    };
    auto ptr = [&](const std::vector<int32_t> & v) -> const void * {
        return v.empty() ? (const void *) &dummy : (const void *) v.data();
    };
    add("mem.key",           GGML_TYPE_I8,  cd.key_dim, n, cd.keys.data());
    add("mem.key_scale",     GGML_TYPE_F32, n, 0, cd.scales.data());
    add("mem.value_offsets", GGML_TYPE_I32, n + 1, 0, cd.offsets.data());
    add("mem.tokens",        GGML_TYPE_I32, (int64_t) cd.tokens.size(), 0, cd.tokens.data());
    add("mem.title_tokens",  GGML_TYPE_I32, std::max<int64_t>(1, (int64_t) cd.title_tokens.size()), 0, ptr(cd.title_tokens));
    add("mem.title_offsets", GGML_TYPE_I32, n + 1, 0, cd.title_offsets.data());
    add("mem.tag_tokens",    GGML_TYPE_I32, std::max<int64_t>(1, (int64_t) cd.tag_tokens.size()), 0, ptr(cd.tag_tokens));
    add("mem.tag_offsets",   GGML_TYPE_I32, n + 1, 0, cd.tag_offsets.data());
    add("mem.pinned",        GGML_TYPE_I32, n, 0, cd.pinned.data());

    const bool ok = gguf_write_to_file(g, path.c_str(), false);
    gguf_free(g);
    ggml_free(ctx);
    if (!ok) {
        err = "gguf write failed: " + path;
    }
    return ok;
}

bool verify_roundtrip(const std::string & path, const card_data & cd, std::string & err) {
    mem_db db;
    if (!mem_load_db(path, db, err)) {
        return false;
    }
    if (db.count != cd.count() || db.key_dim != cd.key_dim ||
        db.tokens.size() != cd.tokens.size() || db.keys_i8 != cd.keys ||
        db.tok_offsets != cd.offsets) {
        err = "card round-trip mismatch: " + path;
        return false;
    }
    return true;
}

bool replace_file(const std::string & tmp, const std::string & dst, std::string & err) {
#ifdef _WIN32
    if (MoveFileExA(tmp.c_str(), dst.c_str(), MOVEFILE_REPLACE_EXISTING)) {
        return true;
    }
    err = "cannot replace card (file in use?): " + dst;
    return false;
#else
    if (std::rename(tmp.c_str(), dst.c_str()) == 0) {
        return true;
    }
    err = "cannot replace card: " + dst;
    return false;
#endif
}

} // namespace

bool mem_card_writer::build(const std::string & path,
                            const std::vector<mem_card_entry> & entries,
                            const llama_vocab * vocab, encode_fn encode,
                            const std::string & name, const std::string & doc_prefix,
                            std::string & err) {
    if (entries.empty()) {
        err = "no entries to build";
        return false;
    }
    card_data cd;
    cd.name       = name.empty() ? "memory" : name;
    cd.doc_prefix = doc_prefix.empty() ? cd.doc_prefix : doc_prefix;

    std::vector<float> probe;
    if (!encode(index_text(entries[0]), probe) || probe.empty()) {
        err = "encoder probe failed";
        return false;
    }
    cd.key_dim = (int32_t) probe.size();

    if (!add_entries(cd, entries, vocab, encode, err)) {
        return false;
    }
    if (!write_gguf(path, cd, err) || !verify_roundtrip(path, cd, err)) {
        return false;
    }
    return true;
}

bool mem_card_writer::append(const std::string & old_path, const std::string & out_path,
                             const std::vector<mem_card_entry> & entries,
                             const llama_vocab * vocab, encode_fn encode,
                             std::string & err) {
    if (entries.empty()) {
        err = "no entries to append";
        return false;
    }
    card_data cd;
    if (!load_old(old_path, cd, err)) {
        return false;
    }
    if (!add_entries(cd, entries, vocab, encode, err)) {
        return false;
    }
    if (!write_gguf(out_path, cd, err) || !verify_roundtrip(out_path, cd, err)) {
        return false;
    }
    return true;
}

bool mem_card_writer::publish(const std::string & path,
                              const std::vector<mem_card_entry> & entries,
                              const llama_vocab * vocab, encode_fn encode,
                              const std::string & name, const std::string & doc_prefix,
                              const std::string * old_path,
                              std::string & err) {
    const std::string tmp = path + ".tmp";
    bool ok = false;
    if (old_path != nullptr) {
        ok = append(*old_path, tmp, entries, vocab, encode, err);
    } else {
        ok = build(tmp, entries, vocab, encode, name, doc_prefix, err);
    }
    if (!ok) {
        std::remove(tmp.c_str());
        return false;
    }
    if (!replace_file(tmp, path, err)) {
        std::remove(tmp.c_str());
        return false;
    }
    return true;
}
