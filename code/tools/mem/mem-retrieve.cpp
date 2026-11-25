#include "mem-internal.h"

#include <algorithm>

float mem_dot(const float * a, const float * b, int64_t n) {
    double s = 0.0;
    for (int64_t i = 0; i < n; i++) {
        s += (double) a[i] * b[i];
    }
    return (float) s;
}

float mem_dot_i8(const int8_t * a, const float * b, int64_t n, float scale) {
    double s = 0.0;
    for (int64_t i = 0; i < n; i++) {
        s += (double) a[i] * b[i];
    }
    return (float) (s * scale);
}

std::vector<mem_pick> mem_pick_top(const std::vector<const mem_db *> & dbs, const float * q,
                                   float threshold, int32_t top_k) {
    std::vector<mem_pick> hits;
    for (int32_t d = 0; d < (int32_t) dbs.size(); d++) {
        const mem_db & db = *dbs[d];
        const bool q8 = db.quantized();
        for (int32_t e = 0; e < db.count; e++) {
            const float score = q8
                ? mem_dot_i8(db.keys_i8.data() + (size_t) e * db.key_dim, q, db.key_dim, db.key_scale[e])
                : mem_dot(db.keys.data() + (size_t) e * db.key_dim, q, db.key_dim);
            if (score > threshold) {
                hits.push_back({ d, e, score });
            }
        }
    }
    std::sort(hits.begin(), hits.end(), [](const mem_pick & a, const mem_pick & b) {
        return a.score > b.score;
    });
    if (top_k > 0 && (int32_t) hits.size() > top_k) {
        hits.resize(top_k);
    }
    return hits;
}
