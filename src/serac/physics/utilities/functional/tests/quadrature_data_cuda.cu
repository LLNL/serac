// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>

#include "serac/serac_config.hpp"
#include "serac/numerics/mesh_utils_base.hpp"

#include "serac/physics/utilities/quadrature_data.hpp"

#include "serac/physics/utilities/functional/functional.hpp"

using namespace serac;

template <typename T>
__global__ void fill(QuadratureData<T>& output, int num_elements, int num_quadrature_points)
{
  int elem_id = threadIdx.x + blockIdx.x * blockDim.x;
  int quad_id = threadIdx.y;
  if (elem_id < num_elements && quad_id < num_quadrature_points) {
    output(elem_id, quad_id) = elem_id * elem_id + quad_id;
  }
}

template <typename T>
__global__ void copy(QuadratureData<T>& destination, QuadratureData<T>& source, int num_elements,
                     int num_quadrature_points)
{
  int elem_id = threadIdx.x + blockIdx.x * blockDim.x;
  int quad_id = threadIdx.y;
  if (elem_id < num_elements && quad_id < num_quadrature_points) {
    destination(elem_id, quad_id) = source(elem_id, quad_id);
  }
}

TEST(QuadratureDataCUDA, basic_fill_and_copy)
{
  constexpr auto mesh_file          = SERAC_REPO_DIR "/data/meshes/star.mesh";
  auto           mesh               = serac::mesh::refineAndDistribute(serac::buildMeshFromFile(mesh_file), 0, 0);
  constexpr int  p                  = 1;
  constexpr int  elements_per_block = 16;
  const int      num_elements       = mesh->GetNE();

  // FIXME: This assumes a homogeneous mesh
  const int geom                  = mesh->GetElementBaseGeometry(0);
  const int num_quadrature_points = mfem::IntRules.Get(geom, p).GetNPoints();

  QuadratureData<int> source(*mesh, p);
  QuadratureData<int> destination(*mesh, p);

  dim3 blocks{elements_per_block, static_cast<unsigned int>(num_quadrature_points)};
  dim3 grids{static_cast<unsigned int>(num_elements / elements_per_block)};

  fill<<<grids, blocks>>>(source, num_elements, num_quadrature_points);
  copy<<<grids, blocks>>>(destination, source, num_elements, num_quadrature_points);
  cudaDeviceSynchronize();

  EXPECT_TRUE(std::equal(source.begin(), source.end(), destination.begin()));
}

struct State {
  double x;
};

bool operator==(const State& lhs, const State& rhs) { return lhs.x == rhs.x; }

class QuadratureDataGPUTest : public ::testing::Test {
protected:
  void SetUp() override
  {
    constexpr auto mesh_file = SERAC_REPO_DIR "/data/meshes/star.mesh";
    default_mesh             = mesh::refineAndDistribute(buildMeshFromFile(mesh_file), 0, 0);
    resetWithNewMesh(*default_mesh);
  }

  void resetWithNewMesh(mfem::ParMesh& new_mesh)
  {
    mesh                = &new_mesh;
    festate             = std::make_unique<FiniteElementState>(*mesh);
    festate->gridFunc() = 0.0;
    residual = std::make_unique<Functional<test_space(trial_space), gpu_policy>>(&festate->space(), &festate->space());
  }
  static constexpr int p   = 1;
  static constexpr int dim = 2;
  using test_space         = H1<p>;
  using trial_space        = H1<p>;
  std::unique_ptr<mfem::ParMesh>                                   default_mesh;
  mfem::ParMesh*                                                   mesh = nullptr;
  std::unique_ptr<FiniteElementState>                              festate;
  std::unique_ptr<Functional<test_space(trial_space), gpu_policy>> residual;
};

struct basic_state_qfunction {
  template <typename x_t, typename field_t, typename state_t>
  __host__ __device__ auto operator()(x_t&& /* x */, field_t&& u, state_t&& state)
  {
    state.x += 0.1;
    return u;
  }
};

TEST_F(QuadratureDataGPUTest, basic_integrals)
{
  QuadratureData<State> qdata(*mesh, p);
  State                 init{0.1};
  qdata = init;
  residual->AddDomainIntegral(Dimension<dim>{}, basic_state_qfunction{}, *mesh, qdata);

  // If we run through it one time...
  mfem::Vector U(festate->space().TrueVSize());
  (*residual)(U);
  // Then each element of the state should have been incremented accordingly...
  State correct{0.2};
  for (const auto& s : qdata) {
    EXPECT_EQ(s, correct);
  }
}

//------------------------------------------------------------------------------
#include "axom/slic/core/SimpleLogger.hpp"

int main(int argc, char* argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  MPI_Init(&argc, &argv);

  axom::slic::SimpleLogger logger;  // create & initialize test logger, finalized when exiting main scope

  result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}
