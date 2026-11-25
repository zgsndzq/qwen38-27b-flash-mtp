#pragma once

#include "mem-card.h"
#include "mem-journal.h"

#include <functional>
#include <string>
#include <vector>

struct mem_dream_config {
    std::string card_path;
    int32_t days        = 7;    // cold-cache age before consolidation
    int32_t max_turns   = 24;   // turns per dream batch
    size_t  max_chars   = 24000;
    int32_t max_tokens  = 1024; // generation budget for one dream
};

// Idle consolidation: cold journal turns -> 27B extraction -> validated card
// entries -> atomic GGUF publish -> journal marks. Inference itself is owned
// by the caller (the server task queue); this class only prepares the prompt
// and consumes the generated text.
class mem_dreamer {
public:
    using encode_fn = mem_card_writer::encode_fn;

    mem_dreamer(mem_dream_config cfg, mem_journal & journal,
                const llama_vocab * vocab, encode_fn encode);

    // pick a cold batch and build the extraction prompt; false = nothing to do
    bool prepare(int64_t now, std::string & prompt, std::string & err);
    bool pending() const { return !batch.empty(); }
    int32_t max_tokens() const { return cfg.max_tokens; }

    // parse + validate + publish + mark; false keeps the batch for retry
    bool finish(const std::string & output, std::string & err);

    // drop the batch without committing (cancel / foreground preemption)
    void abort();

private:
    mem_dream_config cfg;
    mem_journal & journal;
    const llama_vocab * vocab;
    encode_fn encode;
    std::vector<mem_turn> batch;
    int fail_count = 0;
};
