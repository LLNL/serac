// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file cli.hpp
 *
 * @brief This file contains the all the necessary functions and macros required
 *        for interacting with the command line interface.
 */

#ifndef SERAC_INPUT
#define SERAC_INPUT

#include <string>

#include "axom/datastore.hpp"
#include "axom/inlet.hpp"

namespace serac {

namespace input {

/**
 * @brief Defines command line options and parses the found values.
 *
 * @param[in] rank MPI rank of the current node
 * @return map of all given command line options
 */
std::shared_ptr<axom::inlet::Inlet> initialize(std::shared_ptr<axom::sidre::DataStore> datastore,
                                               const std::string& input_file_path)

}  // namespace input
}  // namespace serac

#endif
