#pragma once

#include "llama.h"

#include <cstdint>
#include <string>
#include <vector>

// One mem.gguf: N entries sharing key_dim and hidden.
struct mem_db {
    std::string name;
    int32_t count     = 0;
    int32_t key_dim   = 0;
    int32_t hidden    = 0;
    int32_t version   = 1;
    float   threshold = 0.68f;
    std::string entries_json; // mem.entries metadata, kept for card append
    std::string doc_prefix;   // mem.doc_prefix, kept for card append
    std::vector<float>       keys;        // v1: [count * key_dim]
    std::vector<int8_t>      keys_i8;     // v2: [count * key_dim], per-entry scale
    std::vector<float>       key_scale;   // v2: [count]
    std::vector<llama_token> tokens;      // [n_tokens]
    std::vector<int32_t>     tok_offsets; // [count + 1]
    std::vector<llama_token> title_tokens;   // v2 optional
    std::vector<int32_t>     title_offsets;
    std::vector<llama_token> tag_tokens;     // v2 optional
    std::vector<int32_t>     tag_offsets;
    std::vector<int32_t>     pinned;      // [count], optional: always injected (test fixture)
    std::vector<int32_t>     sym;         // [n_sym], optional
    std::vector<int32_t>     sym_offsets; // [count + 1], optional

    bool quantized() const { return version >= 2 && (int64_t) keys_i8.size() == (int64_t) count * key_dim; }
};

bool mem_load_db(const std::string & path, mem_db & db, std::string & err);

// Query encoder (embedding tower). Pinned to the CPU so it never takes VRAM.
class mem_encoder {
public:
    ~mem_encoder();
    bool load(const std::string & path, int32_t key_dim, std::string & err);
    bool encode(const std::string & text, std::vector<float> & out); // out: key_dim, L2-normalized

private:
    llama_model       * model = nullptr;
    llama_context     * ctx   = nullptr;
    const llama_vocab * vocab = nullptr;
    int32_t n_ctx   = 2048;
    int32_t key_dim = 256;
};

// Pure helpers, free of llama context state.
float mem_dot(const float * a, const float * b, int64_t n);
// int8 weights against a float query, scaled by the entry's per-vector scale
float mem_dot_i8(const int8_t * a, const float * b, int64_t n, float scale);

struct mem_pick {
    int32_t db    = -1;
    int32_t entry = -1;
    float   score = 0.0f;
};

// entries above the threshold, best first, at most top_k
std::vector<mem_pick> mem_pick_top(const std::vector<const mem_db *> & dbs, const float * q,
                                   float threshold, int32_t top_k);
