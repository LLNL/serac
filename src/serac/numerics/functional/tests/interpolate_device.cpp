// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <fstream>
#include <iostream>

#include "mfem.hpp"

#include <gtest/gtest.h>

#include "axom/slic/core/SimpleLogger.hpp"
#include "serac/infrastructure/input.hpp"
#include "serac/numerics/functional/finite_element.hpp"
#include "serac/serac_config.hpp"
#include "serac/mesh/mesh_utils_base.hpp"
#include "serac/numerics/stdfunction_operator.hpp"
#include "serac/numerics/functional/functional.hpp"
#include "serac/numerics/functional/tensor.hpp"

#include "serac/numerics/functional/tests/check_gradient.hpp"



int main {
    using element_type = serac::finite_element<mfem::Geometry::CUBE, serac::H1<3>>;
    double* U_e; //input buffer, 2D array
    double* U_q; //output buffer, 2D array
    double* dU_dxi_q; //gradient 3D
    // allocate buffers with umpire

    RAJA::forall() .. {
    using local_array_mem_policy = RAJA::cuda_shared_mem;
       RAJA::LocalArray<local_array_mem_policy> shared_buffer;
       // copy input buffer into shared_buffer
        element_type::interpolate()
        // check U_q and gradient dU_dxi_q
    }
    return 0;
}