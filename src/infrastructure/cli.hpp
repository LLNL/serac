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

#ifndef SERAC_CLI
#define SERAC_CLI

#include <string>
#include <unordered_map>

namespace serac {

// Command line functionality
namespace cli {

/**
 * @brief Defines command line options and parses the found values.
 *
 * @param[in] rank MPI rank of the current node
 * @return map of all given command line options
 */
std::shared_ptr<std::unordered_map<std::string, std::string>> defineAndParse(int argc, char* argv[], int rank);

/**
 * @brief Prints all given command line options to the screen.
 *
 * @param[in] cli_opts Given command line options to be printed
 * @param[in] rank     MPI rank of the current node
 */
void printGiven(std::shared_ptr<std::unordered_map<std::string, std::string>> cli_opts, int rank);

}  // namespace cli
}  // namespace serac

#endif
