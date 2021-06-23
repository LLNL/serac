// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file output.hpp
 *
 * @brief This file contains the all the necessary functions for outputting simulation data
 */

#pragma once

#include <string>

#include "mfem.hpp"
#include "axom/sidre.hpp"

/**
 * @brief The output related helper functions and objects
 *
 */
namespace serac::output {

/**
 * @brief The output file languages supported
 */
enum class Language
{
  HDF5,
  JSON,
  YAML
};

/**
 * @brief Outputs simulation data from the datastore to the given file
 *
 * @param[in] datastore Root of the Sidre datastore
 * @param[in] file_name_prefix Prefix used for generating output file name
 * @param[in] data_collection_name Name of the Data Collection stored in Sidre
 * @param[in] time Current simulation time
 * @param[in] language The output language format
 */
void outputFields(const axom::sidre::DataStore& datastore, const std::string& data_collection_name,
                  const std::string& file_name_prefix, double time, const Language language = Language::JSON);

}  // namespace serac::output
