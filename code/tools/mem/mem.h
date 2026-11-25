#pragma once

#include "llama.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Engine-side memory layer: encode the question, retrieve the best entry
// from one or more mem.gguf databases, and inject its tokens as a prompt prefix.
// Inactive unless the LLAMA_MEMORY switch is on and a memory + encoder are given.
class server_memory {
public:
    struct config {
        std::vector<std::string> paths;    // mem.gguf files, loaded in order
        std::string encoder;               // embedding tower gguf
        float       threshold = 0.68f;     // retrieval threshold
        float       margin    = 0.0f;      // min top1-top2 gap, 0 = off
        bool        req_toggle = false;    // honour per-request {"mem": false}
        int32_t     top_k     = 3;         // reserved for P2 top-k
        int32_t     budget    = 4096;      // max injected tokens
        std::string mode      = "auto";    // auto | first | off
        std::string rerank    = "sim+sym"; // reserved for P3
    };

    explicit server_memory(config cfg);
    ~server_memory();

    // load databases and the encoder; false on error with a message
    bool init(const llama_vocab * vocab, std::string & err);

    // gate + retrieve; on hit fills head with the entry's tokens and returns true
    bool maybe_inject(const std::string & question, std::vector<llama_token> & head);

    // true when the server may skip injection for individual requests
    bool req_toggle() const;

    // JSON of what the last call injected (empty when nothing was injected)
    std::string last_report() const;
    void clear_report();

    size_t n_dbs() const;
    size_t n_entries() const;

    // hot-swap the database at path (or add it); used after a dream publish
    bool upsert_db(const std::string & path, std::string & err);

    const llama_vocab * vocab() const;

    // serialized tower access, shared with retrieval
    bool encode(const std::string & text, std::vector<float> & out);

private:
    struct impl;
    std::unique_ptr<impl> p;
};
