#pragma once

#include "llama.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

// One card entry before encoding. meta_json is stored verbatim in mem.entries.
struct mem_card_entry {
    std::string title;
    std::string tags;      // space separated
    std::string body;
    std::string meta_json; // object fragment, e.g. {"kind":"task_result"}
};

// Shared card writer: user CLI and server dreaming produce the same v2 layout.
// encode_fn maps index text to a normalized key vector (the embedding tower).
class mem_card_writer {
public:
    using encode_fn = std::function<bool(const std::string &, std::vector<float> &)>;

    static bool build(const std::string & path,
                      const std::vector<mem_card_entry> & entries,
                      const llama_vocab * vocab, encode_fn encode,
                      const std::string & name, const std::string & doc_prefix,
                      std::string & err);

    // load an existing v2 card, append entries, write a full new snapshot
    static bool append(const std::string & old_path, const std::string & out_path,
                       const std::vector<mem_card_entry> & entries,
                       const llama_vocab * vocab, encode_fn encode,
                       std::string & err);

    // write to path + ".tmp", flush, then replace path atomically
    static bool publish(const std::string & path,
                        const std::vector<mem_card_entry> & entries,
                        const llama_vocab * vocab, encode_fn encode,
                        const std::string & name, const std::string & doc_prefix,
                        const std::string * old_path,
                        std::string & err);
};
