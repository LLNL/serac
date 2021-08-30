// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <fstream>
#include <iostream>

#include "mfem.hpp"

#include "axom/slic/core/SimpleLogger.hpp"
#include "serac/infrastructure/input.hpp"
#include "serac/serac_config.hpp"
#include "serac/numerics/expr_template_ops.hpp"
#include "serac/numerics/mesh_utils_base.hpp"
#include "serac/physics/operators/stdfunction_operator.hpp"
#include "serac/physics/utilities/functional/functional.hpp"
#include "serac/physics/utilities/functional/tensor.hpp"
#include "serac/numerics/assembled_sparse_matrix.hpp"
#include "serac/infrastructure/profiling.hpp"
#include "serac/physics/utilities/functional/tuple.hpp"

using namespace serac;
using namespace serac::profiling;

constexpr int dim = 2;
constexpr int p   = 1;

int main(int argc, char*argv[])
{

int num_procs, myid;
MPI_Init(&argc, &argv);
MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
MPI_Comm_rank(MPI_COMM_WORLD, &myid);

axom::slic::SimpleLogger logger;

serac::profiling::initializeCaliper();

int serial_refinement   = 1;
int parallel_refinement = 0;

std::string meshfile2D = SERAC_REPO_DIR "/data/meshes/star.mesh";
auto        mesh = mesh::refineAndDistribute(buildMeshFromFile(meshfile2D), serial_refinement, parallel_refinement);

// Create standard MFEM bilinear and linear forms on H1
auto                        fec = mfem::H1_FECollection(p, dim);
mfem::ParFiniteElementSpace fespace(mesh.get(), &fec);

// Set a random state to evaluate the residual
mfem::ParGridFunction u_global(&fespace);
u_global.Randomize();

mfem::Vector U(fespace.TrueVSize());
u_global.GetTrueDofs(U);

// Define the types for the test and trial spaces using the function arguments
using test_space  = H1<p>;
using trial_space = H1<p>;

// Construct the new functional object using the known test and trial spaces
Functional<test_space(trial_space)> residual(&fespace, &fespace);

// Set the essential boundaries
mfem::Array<int> ess_bdr(mesh->bdr_attributes.Max());
ess_bdr = 1;

residual.SetEssentialBC(ess_bdr);

// Add the total domain residual term to the functional
residual.AddAreaIntegral(
    [=]([[maybe_unused]] auto x, auto temperature) {
      // get the value and the gradient from the input tuple
      auto [u, du_dx] = temperature;
      auto source     = -1.0;
      auto flux       = 1.0 * du_dx;
      return serac::tuple{source, flux};
    },
    *mesh);

// Initialize the solution vector with the essential boundary values
mfem::Vector solution(U.Size());
solution = 1.0;

mfem::Vector zero_vec(U.Size());
zero_vec = 0.0;

mfem::Operator& grad = residual.GetGradient(solution);

mfem::CG(grad, zero_vec, solution, 1, 50000, 1.0e-6, 1.0e-8);

serac::profiling::terminateCaliper();

MPI_Finalize();
}
