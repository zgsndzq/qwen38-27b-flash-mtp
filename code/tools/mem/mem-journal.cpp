#include "mem-journal.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace {

bool replace_file(const std::string & tmp, const std::string & dst) {
#ifdef _WIN32
    return MoveFileExA(tmp.c_str(), dst.c_str(), MOVEFILE_REPLACE_EXISTING) != 0;
#else
    return std::rename(tmp.c_str(), dst.c_str()) == 0;
#endif
}

uint64_t fnv1a(const std::string & s) {
    uint64_t h = 14695981039346656037ull;
    for (unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ull;
    }
    return h;
}

} // namespace

std::string mem_turn_id(const mem_turn & t) {
    const std::string basis = std::to_string(t.ts) + "\n" + t.session + "\n" + t.user + "\n" + t.assistant;
    char buf[32];
    snprintf(buf, sizeof(buf), "%016llx", (unsigned long long) fnv1a(basis));
    return buf;
}

bool mem_journal::open(const std::string & dir_in, size_t cap, std::string & err) {
    dir = dir_in;
    cap_bytes = cap;
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) {
        err = "cannot create journal dir: " + dir;
        return false;
    }
    const fs::path jpath = fs::path(dir) / "turns.jsonl";
    if (fs::exists(jpath)) {
        std::ifstream ifs(jpath);
        std::string line;
        while (std::getline(ifs, line)) {
            if (line.empty()) {
                continue;
            }
            try {
                const json j = json::parse(line);
                mem_turn t;
                t.id        = j.value("id", "");
                t.ts        = j.value("ts", (int64_t) 0);
                t.session   = j.value("session", "");
                t.user      = j.value("user", "");
                t.assistant = j.value("assistant", "");
                if (!t.id.empty() && t.ts > 0) {
                    turns.push_back(std::move(t));
                }
            } catch (const std::exception &) {
                // a torn last line from a crash is dropped, the rest stays usable
            }
        }
        file_bytes = (size_t) fs::file_size(jpath, ec);
    }
    return load_state(err);
}

bool mem_journal::load_state(std::string & err) {
    const fs::path spath = fs::path(dir) / "state.json";
    if (!fs::exists(spath)) {
        return true;
    }
    try {
        std::ifstream ifs(spath);
        const json j = json::parse(ifs);
        for (const auto & id : j.value("archived", json::array())) {
            archived.insert(id.get<std::string>());
        }
        for (const auto & id : j.value("skipped", json::array())) {
            skipped.insert(id.get<std::string>());
        }
    } catch (const std::exception & e) {
        err = std::string("bad journal state: ") + e.what();
        return false;
    }
    return true;
}

bool mem_journal::save_state(std::string & err) {
    json j;
    j["archived"] = json::array();
    j["skipped"]  = json::array();
    for (const auto & id : archived) { j["archived"].push_back(id); }
    for (const auto & id : skipped)  { j["skipped"].push_back(id); }
    const fs::path tmp = fs::path(dir) / "state.json.tmp";
    {
        std::ofstream ofs(tmp);
        ofs << j.dump();
        ofs.flush();
        if (!ofs) {
            err = "cannot write journal state";
            return false;
        }
    }
    if (!replace_file(tmp.string(), (fs::path(dir) / "state.json").string())) {
        err = "cannot replace journal state";
        return false;
    }
    return true;
}

bool mem_journal::append(const mem_turn & turn_in, std::string & err) {
    if (file_bytes >= cap_bytes) {
        if (!cap_warned) {
            cap_warned = true;
            fprintf(stderr, "mem-journal: disk cap reached (%zu bytes), new turns are not recorded\n", cap_bytes);
        }
        return true; // the cap is a policy, not an error: inference continues
    }
    mem_turn t = turn_in;
    if (t.id.empty()) {
        t.id = mem_turn_id(t);
    }
    json j;
    j["id"]        = t.id;
    j["ts"]        = t.ts;
    j["session"]   = t.session;
    j["user"]      = t.user;
    j["assistant"] = t.assistant;
    const std::string line = j.dump() + "\n";
    const fs::path jpath = fs::path(dir) / "turns.jsonl";
    std::ofstream ofs(jpath, std::ios::app);
    ofs << line;
    ofs.flush();
    if (!ofs) {
        err = "cannot append journal turn";
        return false;
    }
    file_bytes += line.size();
    turns.push_back(std::move(t));
    return true;
}

std::vector<mem_turn> mem_journal::select_cold(int64_t now, int32_t days,
                                               int32_t max_turns, size_t max_chars) const {
    std::vector<mem_turn> out;
    const int64_t cutoff = now - (int64_t) days * 24 * 3600;
    size_t chars = 0;
    for (const auto & t : turns) {
        if (t.ts > cutoff || archived.count(t.id) || skipped.count(t.id)) {
            continue;
        }
        const size_t cost = t.user.size() + t.assistant.size();
        if ((int32_t) out.size() >= max_turns || chars + cost > max_chars) {
            break;
        }
        chars += cost;
        out.push_back(t);
    }
    return out;
}

bool mem_journal::mark_done(const std::vector<std::string> & ids, bool is_archived, std::string & err) {
    for (const auto & id : ids) {
        if (is_archived) {
            archived.insert(id);
        } else {
            skipped.insert(id);
        }
    }
    if (!save_state(err)) {
        return false;
    }
    return compact(err);
}

bool mem_journal::compact(std::string & err) {
    std::vector<mem_turn> keep;
    for (auto & t : turns) {
        if (!archived.count(t.id) && !skipped.count(t.id)) {
            keep.push_back(std::move(t));
        }
    }
    const fs::path jpath = fs::path(dir) / "turns.jsonl";
    const fs::path tmp   = fs::path(dir) / "turns.jsonl.tmp";
    {
        std::ofstream ofs(tmp);
        for (const auto & t : keep) {
            json j;
            j["id"]        = t.id;
            j["ts"]        = t.ts;
            j["session"]   = t.session;
            j["user"]      = t.user;
            j["assistant"] = t.assistant;
            ofs << j.dump() << "\n";
        }
        ofs.flush();
        if (!ofs) {
            err = "cannot compact journal";
            return false;
        }
    }
    if (!replace_file(tmp.string(), jpath.string())) {
        err = "cannot replace journal";
        return false;
    }
    std::error_code ec;
    file_bytes = (size_t) fs::file_size(jpath, ec);
    turns = std::move(keep);
    return true;
}
