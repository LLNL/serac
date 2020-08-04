// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/*!
 *******************************************************************************
 * \file mesh_utils.hpp
 *
 * @brief This file contains helper functions for importing and managing
 *        various mesh objects.
 *******************************************************************************
 */

#ifndef MESH_UTILS
#define MESH_UTILS

#include <memory>

#include "mfem.hpp"

namespace serac {
/**
 *****************************************************************************
 * @brief Constructs an MFEM parallel mesh from a file and refines it
 *
 * This opens and reads an external mesh file and constructs a parallel
 * MFEM ParMesh object. The mesh will be refined both serially and
 * in parallel as requested
 *
 * @param [in] mesh_file The mesh file to open
 * @param [in] ref_serial The number of serial refinements
 * @param [in] ref_parallel The number of parallel refinements
 * @return A shared_ptr containing the constructed and refined parallel mesh object
 *****************************************************************************
 */
std::shared_ptr<mfem::ParMesh> buildParallelMesh(const std::string& mesh_file, const int refine_serial = 0,
                                                 const int refine_parallel = 0, const MPI_Comm = MPI_COMM_WORLD);

}  // namespace serac

#endif
