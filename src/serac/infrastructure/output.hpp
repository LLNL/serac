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
 * @brief The output file formats supported
 */
enum class FileFormat
{
  HDF5,
  JSON,
  YAML
};

/**
 * @brief Outputs simulation summary data from the datastore to the given file
 * only on rank 0
 *
 * @param[in] datastore Root of the Sidre datastore
 * @param[in] data_collection_name Name of the Data Collection stored in Sidre
 * @param[in] language The output language format
 */
void outputSummary(const axom::sidre::DataStore& datastore, const std::string& data_collection_name,
                   const FileFormat language = FileFormat::JSON);

/**
 * @brief Outputs simulation field data from the datastore to the given file
 *
 * @param[in] datastore Root of the Sidre datastore
 * @param[in] data_collection_name Name of the Data Collection stored in Sidre
 * @param[in] time Current simulation time
 * @param[in] language The output language format
 */
void outputFields(const axom::sidre::DataStore& datastore, const std::string& data_collection_name, double time,
                  const FileFormat language = FileFormat::JSON);

}  // namespace serac::output
