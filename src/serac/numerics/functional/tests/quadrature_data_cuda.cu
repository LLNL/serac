// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>

#include "serac/serac_config.hpp"
#include "serac/mesh/mesh_utils_base.hpp"
#include "serac/numerics/quadrature_data.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/numerics/functional/functional.hpp"

using namespace serac;

template <typename T>
__global__ void fill(QuadratureDataView<T> output, int num_elements, int num_quadrature_points)
{
  int elem_id = threadIdx.x + blockIdx.x * blockDim.x;
  int quad_id = threadIdx.y;
  if (elem_id < num_elements && quad_id < num_quadrature_points) {
    output(elem_id, quad_id) = elem_id * elem_id + quad_id;
  }
}

template <typename T>
__global__ void copy(QuadratureDataView<T> destination, QuadratureDataView<T> source, int num_elements,
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

  // Note: This assumes a homogeneous mesh
  const int geom                  = mesh->GetElementBaseGeometry(0);
  const int num_quadrature_points = mfem::IntRules.Get(geom, p).GetNPoints();

  QuadratureData<int> source(*mesh, p);
  QuadratureData<int> destination(*mesh, p);

  dim3 blocks{elements_per_block, static_cast<unsigned int>(num_quadrature_points)};
  dim3 grids{static_cast<unsigned int>(num_elements / elements_per_block)};

  fill<<<grids, blocks>>>(QuadratureDataView{source}, num_elements, num_quadrature_points);
  copy<<<grids, blocks>>>(QuadratureDataView{destination}, QuadratureDataView{source}, num_elements,
                          num_quadrature_points);
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
    residual            = std::make_unique<Functional<test_space(trial_space), ExecutionSpace::GPU>>(&festate->space(),
                                                                                          &festate->space());
  }
  static constexpr int p   = 1;
  static constexpr int dim = 2;
  using test_space         = H1<p>;
  using trial_space        = H1<p>;
  std::unique_ptr<mfem::ParMesh>                                            default_mesh;
  mfem::ParMesh*                                                            mesh = nullptr;
  std::unique_ptr<FiniteElementState>                                       festate;
  std::unique_ptr<Functional<test_space(trial_space), ExecutionSpace::GPU>> residual;
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

struct StateWithDefault {
  double x = 0.5;
};

bool operator==(const StateWithDefault& lhs, const StateWithDefault& rhs) { return lhs.x == rhs.x; }

struct basic_state_with_default_qfunction {
  template <typename x_t, typename field_t, typename state_t>
  __host__ __device__ auto operator()(x_t&& /* x */, field_t&& u, state_t&& state)
  {
    state.x += 0.1;
    return u;
  }
};

TEST_F(QuadratureDataGPUTest, basic_integrals_default)
{
  QuadratureData<StateWithDefault> qdata(*mesh, p);
  residual->AddDomainIntegral(Dimension<dim>{}, basic_state_with_default_qfunction{}, *mesh, qdata);
  // If we run through it one time...
  mfem::Vector U(festate->space().TrueVSize());
  (*residual)(U);
  // Then each element of the state should have been incremented accordingly...
  StateWithDefault correct{0.6};
  const auto&      const_qdata = qdata;
  for (auto& s : const_qdata) {
    EXPECT_EQ(s, correct);
  }
}

struct StateWithMultiFields {
  double x = 0.5;
  double y = 0.3;
};

bool almost_equal(double a, double b) { return std::abs(a - b) < 1e-10; }

bool operator==(const StateWithMultiFields& lhs, const StateWithMultiFields& rhs)
{
  return almost_equal(lhs.x, rhs.x) && almost_equal(lhs.y, rhs.y);
}

struct basic_state_with_multi_fields_qfunction {
  template <typename x_t, typename field_t, typename state_t>
  __host__ __device__ auto operator()(x_t&& /* x */, field_t&& u, state_t&& state)
  {
    state.x += 0.1;
    state.y += 0.7;
    return u;
  }
};

TEST_F(QuadratureDataGPUTest, basic_integrals_multi_fields)
{
  QuadratureData<StateWithMultiFields> qdata(*mesh, p);
  residual->AddDomainIntegral(Dimension<dim>{}, basic_state_with_multi_fields_qfunction{}, *mesh, qdata);
  // If we run through it one time...
  mfem::Vector U(festate->space().TrueVSize());
  (*residual)(U);
  // Then each element of the state should have been incremented accordingly...
  StateWithMultiFields correct{0.6, 1.0};
  for (const auto& s : qdata) {
    EXPECT_EQ(s, correct);
  }
}

TEST_F(QuadratureDataGPUTest, basic_integrals_multi_fields_bulk_assignment)
{
  QuadratureData<StateWithMultiFields> qdata(*mesh, p);
  qdata = StateWithMultiFields{0.7, 0.2};
  residual->AddDomainIntegral(Dimension<dim>{}, basic_state_with_multi_fields_qfunction{}, *mesh, qdata);
  // If we run through it one time...
  mfem::Vector U(festate->space().TrueVSize());
  (*residual)(U);
  // Then each element of the state should have been incremented accordingly...
  StateWithMultiFields correct{0.8, 0.9};
  for (const auto& s : qdata) {
    EXPECT_EQ(s, correct);
  }
}

template <typename T>
class QuadratureDataGPUStateManagerTest : public QuadratureDataGPUTest {
public:
  using value_type                              = typename T::value_type;
  static constexpr value_type     initial_state = T::initial_state;
  __host__ __device__ static void mutate(value_type& v, double other = 0.0) { T::mutate(v, other); }
};

struct MultiFieldWrapper {
  using value_type                              = StateWithMultiFields;
  static constexpr value_type     initial_state = {};
  __host__ __device__ static void mutate(value_type& v, double other = 0.0)
  {
    v.x += (0.1 + other);
    v.y += (0.7 + other);
  }
};

struct IntWrapper {
  using value_type                              = int;
  static constexpr value_type     initial_state = 0;
  __host__ __device__ static void mutate(value_type& v, double other = 0.0)
  {
    v += 4;
    v += static_cast<int>(other * 10);
  }
};

struct ThreeBytes {
  char x[3] = {1, 2, 3};
};

bool operator==(const ThreeBytes& lhs, const ThreeBytes& rhs) { return std::equal(lhs.x, lhs.x + 3, rhs.x); }

struct ThreeBytesWrapper {
  using value_type = ThreeBytes;
  static_assert(sizeof(value_type) == 3);
  static constexpr value_type     initial_state = {};
  __host__ __device__ static void mutate(value_type& v, double other = 0.0)
  {
    v.x[0] = static_cast<char>(v.x[0] + 3 + (other * 10));
    v.x[1] = static_cast<char>(v.x[1] + 2 + (other * 10));
    v.x[2] = static_cast<char>(v.x[2] + 1 + (other * 10));
  }
};

using StateTypes = ::testing::Types<MultiFieldWrapper, IntWrapper, ThreeBytesWrapper>;
// NOTE: The extra comma is here due a clang issue where the variadic macro param is not provided
// so instead, we leave it unspecified/empty
TYPED_TEST_SUITE(QuadratureDataGPUStateManagerTest, StateTypes, );

template <typename wrapper_t>
struct state_manager_qfunction {
  template <typename x_t, typename field_t, typename state_t>
  __host__ __device__ auto operator()(x_t&& /* x */, field_t&& u, state_t&& state)
  {
    wrapper_t::mutate(state);
    return u;
  }
};

template <typename wrapper_t>
struct state_manager_varying_qfunction {
  template <typename x_t, typename field_t, typename state_t>
  __host__ __device__ auto operator()(x_t&& x, field_t&& u, state_t&& state)
  {
    double norm = sqnorm(x);
    wrapper_t::mutate(state, norm);
    mutated_data[idx++] = state;
    return u;
  }
  UnifiedArray<typename wrapper_t::value_type>& mutated_data;
  int                                           idx = 0;
};

TYPED_TEST(QuadratureDataGPUStateManagerTest, basic_integrals_state_manager)
{
  constexpr int cycle        = 0;
  const auto    mutated_once = []() {
    typename TestFixture::value_type result = TestFixture::initial_state;
    TestFixture::mutate(result);
    return result;
  }();
  const auto mutated_twice = []() {
    typename TestFixture::value_type result = TestFixture::initial_state;
    TestFixture::mutate(result);
    TestFixture::mutate(result);
    return result;
  }();

  // First set up the Functional object, run it once to update the state once,
  // then save it
  {
    axom::sidre::DataStore datastore;
    serac::StateManager::initialize(datastore, "qdata_gpu_restart");
    // We need to use "this->" explicitly because we are in a derived class template
    serac::StateManager::setMesh(std::move(this->default_mesh));
    // Can't use auto& here because we're in a template context
    serac::QuadratureData<typename TestFixture::value_type>& qdata =
        serac::StateManager::newQuadratureData<typename TestFixture::value_type>("test_data", this->p);
    qdata = TestFixture::initial_state;

    this->residual->AddDomainIntegral(Dimension<TestFixture::dim>{}, state_manager_qfunction<TestFixture>{},
                                      *this->mesh, qdata);
    // If we run through it one time...
    mfem::Vector U(this->festate->space().TrueVSize());
    (*this->residual)(U);

    for (const auto& s : qdata) {
      EXPECT_EQ(s, mutated_once);
    }
    serac::StateManager::save(0.0, cycle);
    serac::StateManager::reset();
  }

  // Then reload the state to make sure it was synced correctly, and update it again before saving
  {
    axom::sidre::DataStore datastore;
    serac::StateManager::initialize(datastore, "qdata_gpu_restart");
    serac::StateManager::load(cycle);
    // Since the original mesh is dead, use the mesh recovered from the save file to build a new Functional
    this->resetWithNewMesh(serac::StateManager::mesh());
    serac::QuadratureData<typename TestFixture::value_type>& qdata =
        serac::StateManager::newQuadratureData<typename TestFixture::value_type>("test_data", this->p);
    // Make sure the changes from the first increment were propagated through
    for (const auto& s : qdata) {
      EXPECT_EQ(s, mutated_once);
    }

    // Note that the mesh here has been recovered from the save file,
    // same for the qdata (or rather the underlying QuadratureFunction)
    this->residual->AddDomainIntegral(Dimension<TestFixture::dim>{}, state_manager_qfunction<TestFixture>{},
                                      *this->mesh, qdata);
    // Then increment it for the second time
    mfem::Vector U(this->festate->space().TrueVSize());
    (*this->residual)(U);
    // Before saving it again
    serac::StateManager::save(0.1, cycle + 1);
    serac::StateManager::reset();
  }

  // Ordered quadrature point data that is unique (mutated with the point's distance from the origin)
  UnifiedArray<typename TestFixture::value_type> origin_mutated_data;

  // Reload the state again to make sure the same synchronization still happens when the data
  // is read in from a restart
  {
    axom::sidre::DataStore datastore;
    serac::StateManager::initialize(datastore, "qdata_gpu_restart");
    serac::StateManager::load(cycle + 1);
    // Since the original mesh is dead, use the mesh recovered from the save file to build a new Functional
    this->resetWithNewMesh(serac::StateManager::mesh());
    serac::QuadratureData<typename TestFixture::value_type>& qdata =
        serac::StateManager::newQuadratureData<typename TestFixture::value_type>("test_data", this->p);
    // Make sure the changes from the second increment were propagated through
    for (const auto& s : qdata) {
      EXPECT_EQ(s, mutated_twice);
    }

    origin_mutated_data.resize(std::distance(qdata.begin(), qdata.end()));

    // this->residual->AddDomainIntegral(Dimension<TestFixture::dim>{},
    //                                   state_manager_varying_qfunction<TestFixture>{origin_mutated_data}, *this->mesh,
    //                                   qdata);
    // Then mutate it for the third time
    mfem::Vector U(this->festate->space().TrueVSize());
    (*this->residual)(U);
    // Before saving it again
    serac::StateManager::save(0.1, cycle + 2);
    serac::StateManager::reset();
  }

  // Reload the state one more time to make sure order is preserved when reloading - the previous mutation
  // included the distance of the quadrature point from the origin (which is unique)
  {
    axom::sidre::DataStore datastore;
    serac::StateManager::initialize(datastore, "qdata_gpu_restart");
    serac::StateManager::load(cycle + 2);
    serac::QuadratureData<typename TestFixture::value_type>& qdata =
        serac::StateManager::newQuadratureData<typename TestFixture::value_type>("test_data", this->p);
    // Make sure the changes from the distance-specified increment were propagated through and in the correct order
    std::size_t i = 0;
    for (const auto& s : qdata) {
      // EXPECT_EQ(s, origin_mutated_data[i]);
      i++;
    }
    serac::StateManager::reset();
  }
}

//------------------------------------------------------------------------------
#include "axom/slic/core/SimpleLogger.hpp"

int main(int argc, char* argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  MPI_Init(&argc, &argv);

  serac::accelerator::initializeDevice();

  axom::slic::SimpleLogger logger;  // create & initialize test logger, finalized when exiting main scope

  result = RUN_ALL_TESTS();

  MPI_Finalize();

  // why does this test need to call terminateDevice,
  // but none of the other CUDA tests do?
  serac::accelerator::terminateDevice();

  return result;
}
