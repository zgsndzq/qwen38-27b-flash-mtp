#include "features.h"

#include "log.h"

#include <fstream>
#include <map>
#include <stdexcept>
#include <utility>

namespace {

// "GGML_CUDA_FA_SHORTQ_MMA" -> an environment variable, "spec-draft-sampling" or
// "sampler_fast_topk" -> an argument name
bool is_env_name(const std::string & key) {
    if (key.empty() || !(key[0] >= 'A' && key[0] <= 'Z')) {
        return false;
    }
    for (char c : key) {
        const bool ok = (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
        if (!ok) {
            return false;
        }
    }
    return true;
}

std::string strip(const std::string & s) {
    const auto b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) {
        return "";
    }
    return s.substr(b, s.find_last_not_of(" \t\r\n") - b + 1);
}

std::string rm_dashes(const std::string & s) {
    size_t i = 0;
    while (i < s.size() && s[i] == '-') {
        ++i;
    }
    return s.substr(i);
}

// Flat INI: one "key = value" per line, '#' or ';' starts a comment, and a
// "[section]" header starts a block of keys that only one model applies.
struct feature_line {
    std::string section;
    std::string key;
    std::string value;
};

std::vector<feature_line> read_ini(const std::string & path) {
    std::ifstream file(path);
    if (!file.good()) {
        throw std::runtime_error("ada: cannot open feature file: " + path);
    }

    std::vector<feature_line> out;
    std::string section;
    std::string line;
    while (std::getline(file, line)) {
        const auto comment = line.find_first_of("#;");
        if (comment != std::string::npos) {
            line = line.substr(0, comment);
        }
        line = strip(line);
        if (line.empty()) {
            continue;
        }
        if (line.front() == '[') {
            const auto close = line.find(']');
            section = close == std::string::npos ? strip(line.substr(1))
                                                 : strip(line.substr(1, close - 1));
            continue;
        }

        const auto eq = line.find('=');
        if (eq == std::string::npos) {
            LOG_WRN("ada: feature file, line without '=' ignored: %s\n", line.c_str());
            continue;
        }

        const std::string key = strip(line.substr(0, eq));
        const std::string val = strip(line.substr(eq + 1));
        if (key.empty()) {
            LOG_WRN("ada: feature file, empty key ignored\n");
            continue;
        }
        out.push_back({section, key, val});
    }
    return out;
}

} // namespace

std::string common_params_find_features_file(int argc, char ** argv) {
    const std::string eq_form = "--features-file=";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i] == nullptr ? "" : argv[i];
        if (arg.compare(0, eq_form.size(), eq_form) == 0) {
            return arg.substr(eq_form.size());
        }
        if (arg == "--features-file" && i + 1 < argc && argv[i + 1] != nullptr) {
            return argv[i + 1];
        }
    }
    return "";
}

int common_params_apply_features_file(const std::string & path,
                                      const std::string & section,
                                      common_params & params,
                                      const std::vector<common_arg> & options) {
    // an option can be addressed by any of its names: its environment variable or
    // any of its argument forms
    std::map<std::string, const common_arg *> key_to_opt;
    for (const auto & opt : options) {
        for (const auto & env : opt.get_env()) {
            key_to_opt[env] = &opt;
        }
        for (const auto & arg : opt.get_args()) {
            key_to_opt[rm_dashes(arg)] = &opt;
        }
    }

    int applied = 0;
    int ignored = 0;
    int skipped = 0;

    for (const auto & e : read_ini(path)) {
        // three tiers: no header or [common] applies to every model, a named
        // header applies only when --features-section names it
        if (!e.section.empty() && e.section != "common" && e.section != section) {
            ++skipped;
            continue;
        }
        const std::string & key   = e.key;
        const std::string & value = e.value;
        try {
            if (is_env_name(key) && key_to_opt.find(key) == key_to_opt.end()) {
                common_set_env(key, value);
                LOG_INF("ada: feature env   %s = %s\n", key.c_str(), value.c_str());
                ++applied;
                continue;
            }

            const auto it = key_to_opt.find(rm_dashes(key));
            if (it == key_to_opt.end()) {
                LOG_WRN("ada: feature '%s' is not known to this build, ignored\n", key.c_str());
                ++ignored;
                continue;
            }

            const common_arg & opt = *it->second;

            if (opt.handler_string) {
                opt.handler_string(params, value);
            } else if (opt.handler_int) {
                opt.handler_int(params, std::stoi(value));
            } else if (opt.handler_bool) {
                opt.handler_bool(params, common_arg_utils::is_truthy(value));
            } else if (opt.handler_void) {
                // a bare switch has no value, so the value decides whether it is on
                if (!common_arg_utils::is_truthy(value)) {
                    LOG_INF("ada: feature off   %s\n", key.c_str());
                    continue;
                }
                opt.handler_void(params);
            } else {
                LOG_WRN("ada: feature '%s' has no usable handler, ignored\n", key.c_str());
                ++ignored;
                continue;
            }

            LOG_INF("ada: feature set   %s = %s\n", key.c_str(), value.c_str());
            ++applied;
        } catch (const std::exception & e) {
            LOG_WRN("ada: feature '%s' = '%s' could not be applied (%s), ignored\n",
                    key.c_str(), value.c_str(), e.what());
            ++ignored;
        }
    }

    LOG_INF("ada: feature file %s [section=%s] -> %d applied, %d ignored, %d skipped\n",
            path.c_str(), section.empty() ? "-" : section.c_str(), applied, ignored, skipped);
    return applied;
}

std::string common_params_find_features_section(int argc, char ** argv) {
    const std::string eq_form = "--features-section=";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i] == nullptr ? "" : argv[i];
        if (arg.compare(0, eq_form.size(), eq_form) == 0) {
            return arg.substr(eq_form.size());
        }
        if (arg == "--features-section" && i + 1 < argc && argv[i + 1] != nullptr) {
            return argv[i + 1];
        }
    }
    return "";
}
