#pragma once

// Server-side conversation state: the llama.cpp equivalent of the OpenAI Responses API
// "previous_response_id" contract.
//
// Why this exists: the chat template re-renders the assistant turn on every request, and its
// rendering of a thinking block is not byte-identical to the tokens the model actually emitted
// (the template inserts a newline the model never generated, and it trims what it replays).
// A single whitespace token is enough to break server_tokens::get_common_prefix(), which then
// discards the entire prompt cache for that turn -- the "second turn refills from scratch" symptom.
//
// So instead of trusting a re-render, the server keeps the exact token sequence it decoded and the
// next request appends only the new messages to it. The client just echoes back the response id,
// exactly like OpenAI's stateful Responses API. Anything that does not match (edited history, a
// different conversation, a missing/unknown id) falls back to the plain full-re-render path, so
// this is a pure opt-in optimization.

#include "server-common.h"

#include <algorithm>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

struct server_session {
    llama_tokens tokens;     // every token the model has decoded for this conversation
    json         messages;   // the `messages` array this conversation was last asked for
    int32_t      n_consumed; // index of the first message NOT covered by `tokens` any more
};

// Process-wide store. llama-server keeps conversations in memory only; after a restart the first
// turn of each conversation simply degrades to a full prompt, which is exactly today's behaviour.
class server_session_store {
public:
    static constexpr size_t max_sessions = 32;

    void put(const std::string & id, const llama_tokens & tokens, const json & messages, int32_t n_consumed) {
        if (id.empty() || tokens.empty() || !messages.is_array() || messages.empty() || n_consumed <= 0) {
            return;
        }

        std::lock_guard<std::mutex> lock(mutex);

        const auto it = std::find(order.begin(), order.end(), id);
        if (it != order.end()) {
            order.erase(it);
        }
        order.push_back(id);

        sessions[id] = server_session{ tokens, messages, n_consumed };

        while (order.size() > max_sessions) {
            sessions.erase(order.front());
            order.erase(order.begin());
        }
    }

    bool get(const std::string & id, llama_tokens & tokens, json & messages, int32_t & n_consumed) const {
        std::lock_guard<std::mutex> lock(mutex);

        const auto it = sessions.find(id);
        if (it == sessions.end()) {
            return false;
        }

        tokens     = it->second.tokens;
        messages   = it->second.messages;
        n_consumed = it->second.n_consumed;
        return true;
    }

    void forget(const std::string & id) {
        std::lock_guard<std::mutex> lock(mutex);

        sessions.erase(id);

        const auto it = std::find(order.begin(), order.end(), id);
        if (it != order.end()) {
            order.erase(it);
        }
    }

private:
    mutable std::mutex mutex;

    std::unordered_map<std::string, server_session> sessions;
    std::vector<std::string> order; // insertion order, oldest first (cheap LRU)
};

inline server_session_store & server_sessions() {
    static server_session_store store;
    return store;
}
