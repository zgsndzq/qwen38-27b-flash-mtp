#pragma once

//
// ada - feature file
//
// Our fork adds switches on top of upstream: an extra sampling mode, a different
// flash-attention routing, a tiered KV cache, and a few environment switches that
// the CUDA backend reads directly. Keeping them in the launch script made the
// script unreadable and made every upstream merge a hunt for "which knob did this
// file set again". They now live in one INI file that the engine reads itself.
//
// Two kinds of keys:
//   * shaped like an environment variable (all upper case, digits and underscore)
//     -> exported with common_set_env()
//   * anything else -> an argument name without its leading dashes, looked up in
//     the normal option table, e.g. "spec-draft-sampling = probabilistic"
//

#include "arg.h"
#include "common.h"

#include <string>
#include <vector>

// Apply the feature file. Returns the number of switches that were applied.
// Unknown keys and malformed values are reported and skipped, never fatal - a
// stale line must not be able to stop the server from starting.
int common_params_apply_features_file(const std::string & path,
                                      common_params & params,
                                      const std::vector<common_arg> & options);

// Pull "--features-file <path>" or "--features-file=<path>" out of argv. The file
// has to be known before the command line is parsed, so that explicit arguments
// still override whatever the file says.
std::string common_params_find_features_file(int argc, char ** argv);
