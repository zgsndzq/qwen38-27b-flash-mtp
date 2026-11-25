#include "mem.h"

#include "mem-internal.h"

#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

namespace {

// tokenize a plain string with the target model's vocab
void append_text(const llama_vocab * vocab, const std::string & s, std::vector<llama_token> & out) {
    if (vocab == nullptr || s.empty()) {
        return;
    }
    std::vector<llama_token> buf(s.size() + 8);
    int n = llama_tokenize(vocab, s.c_str(), (int32_t) s.size(), buf.data(), (int32_t) buf.size(), false, true);
    if (n < 0) {
        buf.resize((size_t) -n);
        n = llama_tokenize(vocab, s.c_str(), (int32_t) s.size(), buf.data(), (int32_t) buf.size(), false, true);
    }
    if (n > 0) {
        out.insert(out.end(), buf.begin(), buf.begin() + n);
    }
}

std::string detok(const llama_vocab * vocab, const std::vector<llama_token> & ids, int32_t begin, int32_t end) {
    std::string s;
    char piece[256];
    for (int32_t i = begin; i < end && vocab != nullptr; i++) {
        int n = llama_token_to_piece(vocab, ids[i], piece, (int32_t) sizeof(piece), 0, true);
        if (n > 0) {
            s.append(piece, n);
        }
    }
    return s;
}

} // namespace

struct server_memory::impl {
    server_memory::config cfg;
    std::vector<std::unique_ptr<mem_db>> dbs;
    std::vector<const mem_db *>          views;
    std::vector<std::string>             db_paths;
    mem_encoder                          encoder;
    const llama_vocab *                  vocab = nullptr;
    std::string                          last;   // JSON report of the last injection
    std::mutex                           mtx;    // guards encoder, views and report
};

server_memory::server_memory(config cfg) : p(new impl) {
    p->cfg = std::move(cfg);
}

server_memory::~server_memory() = default;

bool server_memory::init(const llama_vocab * vocab, std::string & err) {
    p->vocab = vocab;
    for (const auto & path : p->cfg.paths) {
        auto db = std::make_unique<mem_db>();
        if (!mem_load_db(path, *db, err)) {
            return false;
        }
        p->views.push_back(db.get());
        p->dbs.push_back(std::move(db));
        p->db_paths.push_back(path);
    }
    if (p->views.empty()) {
        err = "memory enabled but no database was loaded";
        return false;
    }
    const int32_t key_dim = p->views[0]->key_dim;
    if (!p->encoder.load(p->cfg.encoder, key_dim, err)) {
        return false;
    }
    return true;
}

bool server_memory::maybe_inject(const std::string & question, std::vector<llama_token> & head) {
    std::lock_guard<std::mutex> lock(p->mtx);
    if (p->cfg.mode == "off" || question.empty()) {
        return false;
    }

    std::vector<float> q;
    if (!p->encoder.encode(question, q)) {
        return false;
    }

    const std::vector<mem_pick> picks = mem_pick_top(p->views, q.data(), p->cfg.threshold,
                                                     p->cfg.top_k);
    // pinned entries are carried even when nothing matches, so the model can consult or ignore them
    bool has_pinned = false;
    for (const auto & dbp : p->dbs) {
        for (int32_t e = 0; e < (int32_t) dbp->pinned.size(); e++) {
            if (dbp->pinned[e] != 0) {
                has_pinned = true;
            }
        }
    }
    if (picks.empty() && !has_pinned) {
        return false;
    }
    // optional gap gate: a flat top means the query does not really match anything
    if (!picks.empty() && p->cfg.margin > 0.0f && picks.size() > 1 && picks[0].score - picks[1].score < p->cfg.margin) {
        fprintf(stderr, "memory: skip, top gap %.4f < %.4f\n",
                picks[0].score - picks[1].score, p->cfg.margin);
        return false;
    }

    // wrap in a <knowledge> block so the model treats it as reference material, not as the question
    head.clear();
    append_text(p->vocab,
        "<knowledge source=\"local-memory\" note=\"以下是从本地知识库检索到的资料，"
        "可能与当前问题无关；请按需参考，不要照抄，与问题无关就忽略\">\n",
        head);

    int64_t remaining = p->cfg.budget > 0 ? (int64_t) p->cfg.budget : (int64_t) 1 << 40;
    int64_t injected = 0;
    int32_t used = 0;
    struct rep_t { std::string title; int32_t idx; float score; int32_t tokens; bool pinned; };
    std::vector<rep_t> rep;

    // pinned entries go in regardless of the score: test fixtures the model may consult or ignore
    for (const auto & dbp : p->dbs) {
        const mem_db & db = *dbp;
        if (db.pinned.empty()) {
            continue;
        }
        for (int32_t e = 0; e < db.count && e < (int32_t) db.pinned.size(); e++) {
            if (db.pinned[e] == 0) {
                continue;
            }
            const int32_t begin = db.tok_offsets[e];
            const int32_t end   = db.tok_offsets[e + 1];
            const int32_t n = end - begin;
            if (n <= 0 || (int64_t) n > remaining) {
                continue;
            }
            std::string open = "\n<entry id=\"" + std::to_string(used + 1) + "\"";
            if (!db.title_offsets.empty() && (size_t) e + 1 < db.title_offsets.size()) {
                open += " title=\"" + detok(p->vocab, db.title_tokens,
                                            db.title_offsets[e], db.title_offsets[e + 1]) + "\"";
            }
            open += " pinned=\"true\">\n";
            append_text(p->vocab, open, head);
            head.insert(head.end(), db.tokens.begin() + begin, db.tokens.begin() + end);
            append_text(p->vocab, "\n</entry>\n", head);
            remaining -= n;
            injected += n;
            used++;
            rep.push_back({ detok(p->vocab, db.title_tokens,
                                  db.title_offsets.empty() ? 0 : db.title_offsets[e],
                                  db.title_offsets.empty() ? 0 : db.title_offsets[e + 1]),
                            e, -1.0f, n, true });
            fprintf(stderr, "memory: pinned %s[%d] tokens=%d\n", db.name.c_str(), e, n);
        }
    }

    for (const mem_pick & pick : picks) {
        const mem_db & db = *p->views[pick.db];
        // already carried as pinned: do not inject twice
        if ((size_t) pick.entry < db.pinned.size() && db.pinned[pick.entry] != 0) {
            continue;
        }
        const int32_t begin = db.tok_offsets[pick.entry];
        const int32_t end   = db.tok_offsets[pick.entry + 1];
        const int32_t n = end - begin;
        if (n <= 0 || (int64_t) n > remaining) {
            continue;
        }
        std::string open = "\n<entry id=\"" + std::to_string(used + 1) + "\"";
        if (!db.title_offsets.empty() && (size_t) pick.entry + 1 < db.title_offsets.size()) {
            open += " title=\"" + detok(p->vocab, db.title_tokens,
                                        db.title_offsets[pick.entry], db.title_offsets[pick.entry + 1]) + "\"";
        }
        open += ">\n";
        append_text(p->vocab, open, head);
        head.insert(head.end(), db.tokens.begin() + begin, db.tokens.begin() + end);
        append_text(p->vocab, "\n</entry>\n", head);
        remaining -= n;
        injected += n;
        used++;
        rep.push_back({ detok(p->vocab, db.title_tokens,
                              db.title_offsets.empty() ? 0 : db.title_offsets[pick.entry],
                              db.title_offsets.empty() ? 0 : db.title_offsets[pick.entry + 1]),
                        pick.entry, pick.score, n, false });
        fprintf(stderr, "memory: hit %s[%d] score=%.4f tokens=%d\n",
                db.name.c_str(), pick.entry, pick.score, n);
    }
    if (used == 0) {
        head.clear();
        p->last.clear();
        return false;
    }
    append_text(p->vocab, "</knowledge>\n\n", head);
    fprintf(stderr, "memory: injected %d entr%s, %lld tokens\n", used, used > 1 ? "ies" : "y",
            (long long) injected);

    // report so the client can show what was injected (fork debug aid, /mem proxy uses it)
    // JSON escape + UTF-8 sanitize: detok can produce a broken multi-byte tail,
    // and an invalid UTF-8 byte makes the whole report unparseable on the client side
    auto esc = [](const std::string & s) {
        std::string o;
        char buf[8];
        size_t i = 0, n = s.size();
        while (i < n) {
            const unsigned char c = (unsigned char) s[i];
            if (c == '"' || c == '\\') { o += '\\'; o += (char) c; i++; continue; }
            if (c == '\n') { o += "\\n"; i++; continue; }
            if (c == '\t') { o += "\\t"; i++; continue; }
            if (c < 0x20) { snprintf(buf, sizeof(buf), "\\u%04x", c); o += buf; i++; continue; }
            if (c < 0x80) { o += (char) c; i++; continue; }
            size_t len = (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 0;
            if (len == 0 || i + len > n) { i++; continue; }          // bogus lead byte: drop
            bool okseq = true;
            for (size_t k = 1; k < len; k++) {
                if (((unsigned char) s[i + k] >> 6) != 0x2) { okseq = false; break; }
            }
            if (!okseq) { i++; continue; }                            // broken sequence: drop
            o.append(s, i, len);
            i += len;
        }
        return o;
    };
    std::string js = "{\"tokens\":" + std::to_string(injected) + ",\"entries\":[";
    for (size_t i = 0; i < rep.size(); i++) {
        if (i) js += ",";
        js += "{\"title\":\"" + esc(rep[i].title) + "\",\"idx\":" + std::to_string(rep[i].idx)
            + ",\"tokens\":" + std::to_string(rep[i].tokens);
        if (!rep[i].pinned) js += ",\"score\":" + std::to_string(rep[i].score).substr(0, 6);
        if (rep[i].pinned) js += ",\"pinned\":true";
        js += "}";
    }
    js += "]";
    const std::string body = detok(p->vocab, head, 0, (int32_t) head.size());
    js += ",\"preview\":\"" + esc(body) + "\"}";
    p->last = js;
    return true;
}

std::string server_memory::last_report() const {
    std::lock_guard<std::mutex> lock(p->mtx);
    return p->last;
}

void server_memory::clear_report() {
    std::lock_guard<std::mutex> lock(p->mtx);
    p->last.clear();
}

bool server_memory::upsert_db(const std::string & path, std::string & err) {
    auto db = std::make_unique<mem_db>();
    if (!mem_load_db(path, *db, err)) {
        return false;
    }
    std::lock_guard<std::mutex> lock(p->mtx);
    if (!p->views.empty() && db->key_dim != p->views[0]->key_dim) {
        err = "key dim mismatch on reload: " + path;
        return false;
    }
    for (size_t i = 0; i < p->db_paths.size(); i++) {
        if (p->db_paths[i] == path) {
            p->views[i] = db.get();
            p->dbs[i]   = std::move(db);
            return true;
        }
    }
    p->views.push_back(db.get());
    p->dbs.push_back(std::move(db));
    p->db_paths.push_back(path);
    return true;
}

const llama_vocab * server_memory::vocab() const {
    return p->vocab;
}

bool server_memory::encode(const std::string & text, std::vector<float> & out) {
    std::lock_guard<std::mutex> lock(p->mtx);
    return p->encoder.encode(text, out);
}

bool server_memory::req_toggle() const {
    return p->cfg.req_toggle;
}

size_t server_memory::n_dbs() const {
    return p->dbs.size();
}

size_t server_memory::n_entries() const {
    size_t n = 0;
    for (const auto & db : p->dbs) {
        n += (size_t) db->count;
    }
    return n;
}
