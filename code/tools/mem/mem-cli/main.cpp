// llama-mem: user-facing card tool, same v2 format as the engine memory layer.
//
//   llama-mem build   --model M.gguf --encoder tower.gguf --input cards.jsonl --out my.gguf [--name N]
//   llama-mem append  --model M.gguf --encoder tower.gguf --input cards.jsonl --card my.gguf
//   llama-mem inspect --card my.gguf
//   llama-mem verify  --card my.gguf
//
// input JSONL, one entry per line:
//   {"title": "...", "tags": "kw1 kw2", "body": "...", "meta": {...}}

#include "mem.h"
#include "mem-card.h"
#include "mem-dream.h"
#include "mem-internal.h"

#include "llama.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace {

struct options {
    std::string cmd, model, encoder, input, out, card, name = "memory";
};

bool parse_args(int argc, char ** argv, options & o) {
    if (argc < 2) {
        return false;
    }
    o.cmd = argv[1];
    for (int i = 2; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&](std::string & dst) -> bool {
            if (i + 1 >= argc) {
                return false;
            }
            dst = argv[++i];
            return true;
        };
        if      (a == "--model"   && next(o.model))   {}
        else if (a == "--encoder" && next(o.encoder)) {}
        else if (a == "--input"   && next(o.input))   {}
        else if (a == "--out"     && next(o.out))     {}
        else if (a == "--card"    && next(o.card))    {}
        else if (a == "--name"    && next(o.name))    {}
        else {
            return false;
        }
    }
    return true;
}

void usage() {
    fprintf(stderr,
        "usage:\n"
        "  llama-mem build   --model M.gguf --encoder tower.gguf --input cards.jsonl --out my.gguf [--name N]\n"
        "  llama-mem append  --model M.gguf --encoder tower.gguf --input cards.jsonl --card my.gguf\n"
        "  llama-mem inspect --card my.gguf\n"
        "  llama-mem verify  --card my.gguf\n"
        "  llama-mem selftest --model M.gguf --encoder tower.gguf [--out scratch-dir]\n");
}

bool read_entries(const std::string & path, std::vector<mem_card_entry> & entries, std::string & err) {
    std::ifstream ifs(path);
    if (!ifs) {
        err = "cannot open input: " + path;
        return false;
    }
    std::string line;
    int lineno = 0;
    while (std::getline(ifs, line)) {
        lineno++;
        if (line.empty()) {
            continue;
        }
        try {
            const json j = json::parse(line);
            mem_card_entry e;
            e.title = j.value("title", "");
            e.body  = j.value("body", "");
            if (j.contains("tags")) {
                if (j.at("tags").is_array()) {
                    for (const auto & t : j.at("tags")) {
                        if (!e.tags.empty()) { e.tags += " "; }
                        e.tags += t.get<std::string>();
                    }
                } else {
                    e.tags = j.at("tags").get<std::string>();
                }
            }
            if (j.contains("meta")) {
                e.meta_json = j.at("meta").dump();
            }
            if (e.title.empty() || e.body.empty()) {
                err = "line " + std::to_string(lineno) + ": title and body are required";
                return false;
            }
            entries.push_back(std::move(e));
        } catch (const std::exception & ex) {
            err = "line " + std::to_string(lineno) + ": " + ex.what();
            return false;
        }
    }
    if (entries.empty()) {
        err = "input has no entries";
        return false;
    }
    return true;
}

// vocab-only load: the card tool never needs the 27B weights
const llama_vocab * load_vocab(const std::string & path, llama_model ** model_out, std::string & err) {
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.vocab_only   = true;
    llama_model * model = llama_model_load_from_file(path.c_str(), mp);
    if (model == nullptr) {
        err = "cannot load model vocab: " + path;
        return nullptr;
    }
    *model_out = model;
    return llama_model_get_vocab(model);
}

int run_build_append(const options & o, bool append) {
    std::string err;
    std::vector<mem_card_entry> entries;
    if (!read_entries(o.input, entries, err)) {
        fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }
    llama_model * model = nullptr;
    const llama_vocab * vocab = load_vocab(o.model, &model, err);
    if (vocab == nullptr) {
        fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }
    mem_encoder enc;
    if (!enc.load(o.encoder, 256, err)) {
        fprintf(stderr, "error: %s\n", err.c_str());
        llama_model_free(model);
        return 1;
    }
    auto encode = [&](const std::string & text, std::vector<float> & out) {
        return enc.encode(text, out);
    };
    bool ok = false;
    if (append) {
        ok = mem_card_writer::append(o.card, o.card, entries, vocab, encode, err);
    } else {
        ok = mem_card_writer::build(o.out, entries, vocab, encode, o.name,
                                    "title: " + o.name + " | text: ", err);
    }
    llama_model_free(model);
    if (!ok) {
        fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }
    printf("ok: %zu entries -> %s\n", entries.size(), append ? o.card.c_str() : o.out.c_str());
    return 0;
}

int run_inspect_verify(const options & o, bool verify) {
    std::string err;
    mem_db db;
    if (!mem_load_db(o.card, db, err)) {
        fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }
    printf("card: %s\n  name=%s version=%d entries=%d key_dim=%d tokens=%zu\n",
           o.card.c_str(), db.name.c_str(), db.version, db.count, db.key_dim, db.tokens.size());
    if (!db.entries_json.empty()) {
        try {
            const json meta = json::parse(db.entries_json);
            if (meta.is_array()) {
                const size_t n = verify ? meta.size() : std::min<size_t>(meta.size(), 20);
                for (size_t i = 0; i < n; i++) {
                    const auto & m = meta[i];
                    printf("  [%zu] %s (%d tokens)\n", i,
                           m.value("title", "").c_str(), m.value("tokens", 0));
                }
                if (!verify && meta.size() > n) {
                    printf("  ... %zu more\n", meta.size() - n);
                }
            }
        } catch (const std::exception &) {
            printf("  (mem.entries metadata is not valid JSON)\n");
        }
    }
    if (verify) {
        bool ok = db.count > 0 && db.quantized()
            && (int64_t) db.tok_offsets.size() == db.count + 1
            && db.tok_offsets.front() == 0
            && db.tok_offsets.back() == (int32_t) db.tokens.size();
        for (int32_t i = 0; ok && i < db.count; i++) {
            ok = db.tok_offsets[i] <= db.tok_offsets[i + 1];
        }
        printf("%s\n", ok ? "verify: PASS" : "verify: FAIL");
        return ok ? 0 : 1;
    }
    return 0;
}

// end-to-end dream pipeline check without a server: fake cold journal ->
// prepare -> canned model output -> finish -> card + reclaim
int run_selftest(const options & o) {
    std::string err;
    const std::string dir = o.out.empty() ? "mem-selftest" : o.out;
    const std::string card = dir + "/dream.gguf";
    std::filesystem::remove_all(dir);

    llama_model * model = nullptr;
    const llama_vocab * vocab = load_vocab(o.model, &model, err);
    if (vocab == nullptr) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
    mem_encoder enc;
    if (!enc.load(o.encoder, 256, err)) {
        fprintf(stderr, "error: %s\n", err.c_str()); llama_model_free(model); return 1;
    }
    auto encode = [&](const std::string & text, std::vector<float> & out) {
        return enc.encode(text, out);
    };

    int rc = 1;
    {
        mem_journal jrnl;
        if (!jrnl.open(dir, 64 * 1024 * 1024, err)) {
            fprintf(stderr, "error: %s\n", err.c_str()); llama_model_free(model); return 1;
        }
        const int64_t now = (int64_t) time(nullptr);
        for (int i = 0; i < 3; i++) {
            mem_turn t;
            t.ts = now - 8 * 86400 - i;
            t.session = "selftest";
            t.user = "问题 " + std::to_string(i);
            t.assistant = "回答 " + std::to_string(i);
            t.id = mem_turn_id(t);
            if (!jrnl.append(t, err)) {
                fprintf(stderr, "error: %s\n", err.c_str()); llama_model_free(model); return 1;
            }
        }
        mem_dream_config cfg;
        cfg.card_path = card;
        cfg.days = 7;
        mem_dreamer dreamer(cfg, jrnl, vocab, encode);
        std::string prompt;
        if (!dreamer.prepare(now, prompt, err) || prompt.find("问题 0") == std::string::npos) {
            fprintf(stderr, "selftest: prepare failed: %s\n", err.c_str());
            llama_model_free(model); return 1;
        }
        const std::string output =
            "{\"entries\": ["
            "{\"title\": \"自测事实\", \"tags\": \"selftest\", \"body\": \"做梦流水线自测生成的第一条记忆。\", \"kind\": \"fact\"},"
            "{\"title\": \"自测建议\", \"tags\": \"selftest\", \"body\": \"做梦流水线自测生成的第二条记忆。\", \"kind\": \"proposal\"}]}";
        if (!dreamer.finish(output, err)) {
            fprintf(stderr, "selftest: finish failed: %s\n", err.c_str());
            llama_model_free(model); return 1;
        }
        mem_db db;
        if (!mem_load_db(card, db, err) || db.count != 2) {
            fprintf(stderr, "selftest: card check failed: %s\n", err.c_str());
            llama_model_free(model); return 1;
        }
        if (!jrnl.select_cold(now, 7, 24, 24000).empty()) {
            fprintf(stderr, "selftest: cold turns were not reclaimed\n");
            llama_model_free(model); return 1;
        }
        printf("selftest: PASS (journal -> dream -> card[2] -> reclaim)\n");
        rc = 0;
    }
    llama_model_free(model);
    std::filesystem::remove_all(dir);
    return rc;
}

} // namespace

int main(int argc, char ** argv) {
    options o;
    if (!parse_args(argc, argv, o)) {
        usage();
        return 1;
    }
    llama_backend_init();
    int rc = 1;
    if (o.cmd == "build" && !o.model.empty() && !o.encoder.empty() && !o.input.empty() && !o.out.empty()) {
        rc = run_build_append(o, false);
    } else if (o.cmd == "append" && !o.model.empty() && !o.encoder.empty() && !o.input.empty() && !o.card.empty()) {
        rc = run_build_append(o, true);
    } else if (o.cmd == "inspect" && !o.card.empty()) {
        rc = run_inspect_verify(o, false);
    } else if (o.cmd == "verify" && !o.card.empty()) {
        rc = run_inspect_verify(o, true);
    } else if (o.cmd == "selftest" && !o.model.empty() && !o.encoder.empty()) {
        rc = run_selftest(o);
    } else {
        usage();
    }
    llama_backend_free();
    return rc;
}
