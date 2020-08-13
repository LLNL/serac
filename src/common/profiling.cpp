// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "common/profiling.hpp"

#ifdef SERAC_USE_CALIPER
namespace {
cali::ConfigManager mgr;
}  // namespace
#endif

namespace serac::profiling {

void initializeCaliper(const std::string& options)
{
#ifdef SERAC_USE_CALIPER
  mgr.add(options.c_str());
  mgr.add("event-trace");
  mgr.start();
#else
  // Silence warning
  static_cast<void>(options);
#endif
}

void terminateCaliper()
{
#ifdef SERAC_USE_CALIPER
  mgr.flush();
#endif
}

}  // namespace serac::profiling
