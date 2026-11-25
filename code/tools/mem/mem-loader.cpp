#include "mem-internal.h"

#include "ggml.h"

#include <cstring>

namespace {

uint32_t kv_u32(gguf_context * g, const char * key, uint32_t def) {
    const int i = gguf_find_key(g, key);
    return i < 0 ? def : gguf_get_val_u32(g, i);
}

float kv_f32(gguf_context * g, const char * key, float def) {
    const int i = gguf_find_key(g, key);
    return i < 0 ? def : gguf_get_val_f32(g, i);
}

std::string kv_str(gguf_context * g, const char * key, const char * def) {
    const int i = gguf_find_key(g, key);
    return i < 0 ? std::string(def) : std::string(gguf_get_val_str(g, i));
}

// Copy a tensor's raw data into dst. Optional tensors may be missing.
template <typename T>
bool read_tensor(ggml_context * meta, gguf_context * g, const char * name,
                 enum ggml_type type, std::vector<T> & dst, bool required) {
    const int i = gguf_find_tensor(g, name);
    if (i < 0) {
        return !required;
    }
    if (gguf_get_tensor_type(g, i) != type) {
        return false;
    }
    const ggml_tensor * t = ggml_get_tensor(meta, name);
    if (t == nullptr) {
        return false;
    }
    const size_t n = ggml_nelements(t);
    dst.resize(n);
    std::memcpy(dst.data(), t->data, n * sizeof(T));
    return true;
}

} // namespace

bool mem_load_db(const std::string & path, mem_db & db, std::string & err) {
    struct ggml_context * meta = nullptr;
    struct gguf_init_params gp = { /*.no_alloc = */ false, /*.ctx = */ &meta };
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (g == nullptr) {
        err = "cannot open memory database: " + path;
        return false;
    }

    db.name      = kv_str(g, "mem.name", "memory");
    db.count     = (int32_t) kv_u32(g, "mem.count", 0);
    db.key_dim   = (int32_t) kv_u32(g, "mem.key_dim", 0);
    db.hidden    = (int32_t) kv_u32(g, "mem.hidden", 0);
    db.threshold = kv_f32(g, "mem.threshold", 0.68f);
    db.version   = (int32_t) kv_u32(g, "mem.version", 1);
    db.entries_json = kv_str(g, "mem.entries", "");
    db.doc_prefix   = kv_str(g, "mem.doc_prefix", "");

    bool ok = db.count > 0 && db.key_dim > 0
        && read_tensor(meta, g, "mem.tokens",       GGML_TYPE_I32, db.tokens,      true)
        && read_tensor(meta, g, "mem.value_offsets", GGML_TYPE_I32, db.tok_offsets, true);
    if (ok) {
        if (db.version >= 2) {
            // int8 keys + per-entry scale: 256 dims cost 256 B instead of 1 KiB
            ok = read_tensor(meta, g, "mem.key",       GGML_TYPE_I8,  db.keys_i8,   true)
              && read_tensor(meta, g, "mem.key_scale", GGML_TYPE_F32, db.key_scale, true)
              && (int64_t) db.keys_i8.size()   == (int64_t) db.count * db.key_dim
              && (int64_t) db.key_scale.size() == db.count;
            read_tensor(meta, g, "mem.title_tokens", GGML_TYPE_I32, db.title_tokens, false);
            read_tensor(meta, g, "mem.title_offsets", GGML_TYPE_I32, db.title_offsets, false);
            read_tensor(meta, g, "mem.tag_tokens",   GGML_TYPE_I32, db.tag_tokens,   false);
            read_tensor(meta, g, "mem.tag_offsets",  GGML_TYPE_I32, db.tag_offsets,  false);
            read_tensor(meta, g, "mem.pinned",       GGML_TYPE_I32, db.pinned,       false);
        } else {
            ok = read_tensor(meta, g, "mem.key", GGML_TYPE_F32, db.keys, true)
              && (int64_t) db.keys.size() == (int64_t) db.count * db.key_dim;
        }
    }
    if (ok) {
        read_tensor(meta, g, "mem.sym",         GGML_TYPE_I32, db.sym,         false);
        read_tensor(meta, g, "mem.sym_offsets", GGML_TYPE_I32, db.sym_offsets, false);
        ok = (int64_t) db.tok_offsets.size() == db.count + 1;
    }
    if (!ok) {
        err = "bad memory database: " + path;
    }

    gguf_free(g);
    ggml_free(meta);
    return ok;
}
