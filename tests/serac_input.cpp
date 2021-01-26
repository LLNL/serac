// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/infrastructure/input.hpp"

#include <gtest/gtest.h>
#include "mfem.hpp"

class SlicErrorException : public std::exception {
};

class InputTest : public ::testing::Test {
protected:
  static void SetUpTestSuite()
  {
    axom::slic::setAbortFunction([]() { throw SlicErrorException{}; });
    axom::slic::setAbortOnError(true);
    axom::slic::setAbortOnWarning(false);
  }

  void SetUp() override
  {
    // Set up mesh
    auto reader = std::make_unique<axom::inlet::LuaReader>();
    inlet_.emplace(std::move(reader), datastore_.getRoot());
    reader_ = &(inlet_->reader());
  }

  // Initialization cannot occur during construction, has to be initialized later
  axom::inlet::Reader*              reader_ = nullptr;
  std::optional<axom::inlet::Inlet> inlet_;

private:
  axom::sidre::DataStore datastore_;
};

namespace serac {

TEST_F(InputTest, vec_1d)
{
  reader_->parseString("vec = { x = 4.75 }");
  auto& vec_table = inlet_->addTable("vec");
  input::defineVectorInputFileSchema(vec_table);
  auto vec = vec_table.get<mfem::Vector>();
  EXPECT_EQ(vec.Size(), 1);
  EXPECT_FLOAT_EQ(vec(0), 4.75);
}

TEST_F(InputTest, vec_2d)
{
  reader_->parseString("vec = { x = 4.75, y = 6.81 }");
  auto& vec_table = inlet_->addTable("vec");
  input::defineVectorInputFileSchema(vec_table);
  auto vec = vec_table.get<mfem::Vector>();
  EXPECT_EQ(vec.Size(), 2);
  EXPECT_FLOAT_EQ(vec(0), 4.75);
  EXPECT_FLOAT_EQ(vec(1), 6.81);
}

TEST_F(InputTest, vec_3d)
{
  reader_->parseString("vec = { x = 4.75, y = 6.81, z = -8.33 }");
  auto& vec_table = inlet_->addTable("vec");
  input::defineVectorInputFileSchema(vec_table);
  auto vec = vec_table.get<mfem::Vector>();
  EXPECT_EQ(vec.Size(), 3);
  EXPECT_FLOAT_EQ(vec(0), 4.75);
  EXPECT_FLOAT_EQ(vec(1), 6.81);
  EXPECT_FLOAT_EQ(vec(2), -8.33);
}

TEST_F(InputTest, coef_build_scalar)
{
  reader_->parseString("coef_opts = { coef = function(x, y, z) return y * 2 + z end, component = 1}");
  auto& coef_table = inlet_->addTable("coef_opts");
  input::CoefficientInputOptions::defineInputFileSchema(coef_table);
  auto coef_opts = coef_table.get<input::CoefficientInputOptions>();
  EXPECT_EQ(coef_opts.component, 1);
  EXPECT_FALSE(coef_opts.isVector());
  mfem::Vector test_vec(3);
  test_vec(0)                 = 1;
  test_vec(1)                 = 2;
  test_vec(2)                 = 3;
  const auto& func            = std::get<input::CoefficientInputOptions::ScalarFunc>(coef_opts.func);
  auto        expected_result = test_vec(1) * 2 + test_vec(2);
  EXPECT_FLOAT_EQ(func(test_vec), expected_result);
  EXPECT_NO_THROW(coef_opts.constructScalar());
}

TEST_F(InputTest, coef_build_vec_from_scalar)
{
  reader_->parseString("coef_opts = { coef = function(x, y, z) return y * 2 + z end, component = 1}");
  auto& coef_table = inlet_->addTable("coef_opts");
  input::CoefficientInputOptions::defineInputFileSchema(coef_table);
  auto coef_opts = coef_table.get<input::CoefficientInputOptions>();
  EXPECT_THROW(coef_opts.constructVector(), SlicErrorException);
}

TEST_F(InputTest, coef_build_vector)
{
  reader_->parseString("coef_opts = { vec_coef = function(x, y, z) return y * 2, z, x end }");
  auto& coef_table = inlet_->addTable("coef_opts");
  input::CoefficientInputOptions::defineInputFileSchema(coef_table);
  auto coef_opts = coef_table.get<input::CoefficientInputOptions>();
  EXPECT_TRUE(coef_opts.isVector());
  mfem::Vector test_vec(3);
  test_vec(0)       = 1;
  test_vec(1)       = 2;
  test_vec(2)       = 3;
  const auto&  func = std::get<input::CoefficientInputOptions::VecFunc>(coef_opts.func);
  mfem::Vector expected_result(3);
  expected_result(0) = test_vec(1) * 2;
  expected_result(1) = test_vec(2);
  expected_result(2) = test_vec(0);
  mfem::Vector result(3);
  func(test_vec, result);
  for (int i = 0; i < result.Size(); i++) {
    EXPECT_FLOAT_EQ(result[i], expected_result[i]);
  }
  EXPECT_NO_THROW(coef_opts.constructVector());
}

TEST_F(InputTest, coef_build_scalar_from_vec)
{
  reader_->parseString("coef_opts = { vec_coef = function(x, y, z) return y * 2, z, x end }");
  auto& coef_table = inlet_->addTable("coef_opts");
  input::CoefficientInputOptions::defineInputFileSchema(coef_table);
  auto coef_opts = coef_table.get<input::CoefficientInputOptions>();
  EXPECT_THROW(coef_opts.constructScalar(), SlicErrorException);
}

}  // namespace serac

//------------------------------------------------------------------------------
#include "axom/slic/core/SimpleLogger.hpp"

int main(int argc, char* argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  MPI_Init(&argc, &argv);

  axom::slic::SimpleLogger logger;  // create & initialize test logger, finalized when
                                    // exiting main scope

  result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}
