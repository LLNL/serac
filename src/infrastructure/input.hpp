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

#include "axom/inlet.hpp"
#include "axom/sidre.hpp"

namespace serac {

namespace input {

/**
 * @brief Initializes Inlet with the given datastore and input file.
 *
 * @param[in] datastore Root of the Sidre datastore
 * @param[in] input_file_path Path to user given input file
 * @return initialized Inlet instance
 */
std::shared_ptr<axom::inlet::Inlet> initialize(axom::sidre::DataStore& datastore,
                                               const std::string&      input_file_path);

/**
 * @brief Returns the absolute path of the given mesh either relative
 * to CWD or the input file
 *
 * @param[in] mesh_path Given mesh path from input file
 * @param[in] input_file_path Path to user given input file
 * @return initialized Inlet instance
 */
std::string findMeshFilePath(const std::string& mesh_path, const std::string& input_file_path);

/**
 * @brief Returns the absolute directory of the given file path.
 *
 * @param[in] file_path path to a file
 * @return Absolute directory of given file path
 */
std::string fullDirectoryFromPath(const std::string& file_path);

}  // namespace input
}  // namespace serac

#endif
