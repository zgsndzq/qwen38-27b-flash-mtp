#pragma once

#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

// One completed visible turn, the raw material for dreaming.
struct mem_turn {
    std::string id;
    int64_t ts = 0; // unix seconds
    std::string session;
    std::string user;
    std::string assistant;
};

// Append-only turn journal with archived/skipped marks and a disk cap.
// Layout: <dir>/turns.jsonl + <dir>/state.json. Only manages its own directory.
class mem_journal {
public:
    bool open(const std::string & dir, size_t cap_bytes, std::string & err);

    bool append(const mem_turn & turn, std::string & err);

    // unarchived turns older than days, oldest first, within both budgets
    std::vector<mem_turn> select_cold(int64_t now, int32_t days,
                                      int32_t max_turns, size_t max_chars) const;

    // persist marks, then compact marked turns out of the journal
    bool mark_done(const std::vector<std::string> & ids, bool archived, std::string & err);

    size_t size_bytes() const { return file_bytes; }
    size_t n_turns() const { return turns.size(); }

private:
    bool load_state(std::string & err);
    bool save_state(std::string & err);
    bool compact(std::string & err);

    std::string dir;
    size_t cap_bytes = 64 * 1024 * 1024;
    size_t file_bytes = 0;
    bool cap_warned = false;
    std::vector<mem_turn> turns;
    std::unordered_set<std::string> archived;
    std::unordered_set<std::string> skipped;
};

std::string mem_turn_id(const mem_turn & turn);
