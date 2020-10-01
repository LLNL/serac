// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "infrastructure/profiling.hpp"

#include "infrastructure/logger.hpp"

#ifdef SERAC_USE_CALIPER
#include <optional>

namespace {
std::optional<cali::ConfigManager> mgr;
}  // namespace
#endif

namespace serac::profiling {

void initializeCaliper(const std::string& options)
{
#ifdef SERAC_USE_CALIPER
  mgr               = cali::ConfigManager();
  auto check_result = mgr->check(options.c_str());
  if (check_result.empty()) {
    mgr->add(options.c_str());
  } else {
    SLIC_WARNING("Caliper options invalid, ignoring: " << check_result);
  }
  // Defaults, should probably always be enabled
  mgr->add("event-trace, runtime-report");
  mgr->start();
#else
  // Silence warning
  static_cast<void>(options);
#endif
}

void terminateCaliper()
{
#ifdef SERAC_USE_CALIPER
  if (mgr) {
    mgr->stop();
    mgr->flush();
  }
#endif
}

}  // namespace serac::profiling
