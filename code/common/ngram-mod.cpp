#include "ngram-mod.h"

#include <algorithm>
#include <cstring>
#include <fstream>

//
// common_ngram_mod
//

common_ngram_mod::common_ngram_mod(uint16_t n, size_t size) : n(n), used(0) {
    entries.resize(size);

    reset();
}

size_t common_ngram_mod::idx(const entry_t * tokens) const {
    size_t res = 0;

    for (size_t i = 0; i < n; ++i) {
        res = res*6364136223846793005ULL + tokens[i];
    }

    res = res % entries.size();

    return res;
}

void common_ngram_mod::add(const entry_t * tokens) {
    const size_t i = idx(tokens);

    if (entries[i] == EMPTY) {
        used++;
    }

    entries[i] = tokens[n];
}

common_ngram_mod::entry_t common_ngram_mod::get(const entry_t * tokens) const {
    const size_t i = idx(tokens);

    return entries[i];
}

void common_ngram_mod::reset() {
    std::fill(entries.begin(), entries.end(), EMPTY);
    used = 0;
}

size_t common_ngram_mod::get_n() const {
    return n;
}

size_t common_ngram_mod::get_used() const {
    return used;
}

size_t common_ngram_mod::size() const {
    return entries.size();
}

size_t common_ngram_mod::size_bytes() const {
    return entries.size() * sizeof(entries[0]);
}

// the file header: magic, n, table size, used count, format version
static const char NGRAM_MOD_MAGIC[8] = { 'N', 'G', 'M', 'O', 'D', 'C', '1', 0 };

bool common_ngram_mod::save(const std::string & path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        return false;
    }

    const uint32_t hdr[4] = {
        (uint32_t) n,
        (uint32_t) entries.size(),
        (uint32_t) used,
        1u,
    };

    f.write(NGRAM_MOD_MAGIC, sizeof(NGRAM_MOD_MAGIC));
    f.write((const char *) hdr, sizeof(hdr));
    f.write((const char *) entries.data(), (std::streamsize) (entries.size() * sizeof(entry_t)));

    return f.good();
}

bool common_ngram_mod::load(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return false;
    }

    char magic[sizeof(NGRAM_MOD_MAGIC)];
    f.read(magic, sizeof(magic));
    if (f.gcount() != (std::streamsize) sizeof(magic) || memcmp(magic, NGRAM_MOD_MAGIC, sizeof(magic)) != 0) {
        return false;
    }

    uint32_t hdr[4];
    f.read((char *) hdr, sizeof(hdr));
    if (!f) {
        return false;
    }

    if (hdr[0] != (uint32_t) n || hdr[1] != (uint32_t) entries.size() || hdr[3] != 1u) {
        return false;
    }

    f.read((char *) entries.data(), (std::streamsize) (entries.size() * sizeof(entry_t)));
    if (!f) {
        return false;
    }

    // recount instead of trusting the stored value: a truncated or edited file must
    // not inflate the occupancy counter, it drives the reset threshold
    size_t n_used = 0;
    for (size_t i = 0; i < entries.size(); ++i) {
        n_used += (entries[i] != EMPTY);
    }
    used = n_used;

    return true;
}
