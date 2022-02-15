// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file about.hpp
 *
 * @brief This file contains the interface used for retrieving information
 * about how the driver is configured.
 */

#pragma once

#include <string>

namespace serac {

/**
 * @brief Returns a string about the configuration of Serac
 *
 * @return string containing various configuration information about Serac
 */
std::string about();

/**
 * @brief Returns a string for the Git SHA when the driver was built
 *
 * Note: This will not update unless you re-run CMake between commits.
 *
 * @return string value of the Git SHA if built in a Git repo, empty if not
 */
std::string gitSHA();

/**
 * @brief Returns a string for the version of Serac
 *
 * @param[in] add_SHA boolean for whether to add the Git SHA to the version if available
 *
 * @return string value of the version of Serac
 */
std::string version(bool add_SHA = true);

}  // namespace serac
