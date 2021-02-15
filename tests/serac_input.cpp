// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
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
  EXPECT_DOUBLE_EQ(vec(0), 4.75);
}

TEST_F(InputTest, vec_2d)
{
  reader_->parseString("vec = { x = 4.75, y = 6.81 }");
  auto& vec_table = inlet_->addTable("vec");
  input::defineVectorInputFileSchema(vec_table);
  auto vec = vec_table.get<mfem::Vector>();
  EXPECT_EQ(vec.Size(), 2);
  EXPECT_DOUBLE_EQ(vec(0), 4.75);
  EXPECT_DOUBLE_EQ(vec(1), 6.81);
}

TEST_F(InputTest, vec_3d)
{
  reader_->parseString("vec = { x = 4.75, y = 6.81, z = -8.33 }");
  auto& vec_table = inlet_->addTable("vec");
  input::defineVectorInputFileSchema(vec_table);
  auto vec = vec_table.get<mfem::Vector>();
  EXPECT_EQ(vec.Size(), 3);
  EXPECT_DOUBLE_EQ(vec(0), 4.75);
  EXPECT_DOUBLE_EQ(vec(1), 6.81);
  EXPECT_DOUBLE_EQ(vec(2), -8.33);
}

TEST_F(InputTest, coef_build_scalar)
{
  reader_->parseString("coef_opts = { scalar_function = function(v) return v.y * 2 + v.z end, component = 1}");
  auto& coef_table = inlet_->addTable("coef_opts");
  input::CoefficientInputOptions::defineInputFileSchema(coef_table);
  auto coef_opts = coef_table.get<input::CoefficientInputOptions>();
  EXPECT_EQ(coef_opts.component, 1);
  EXPECT_FALSE(coef_opts.isVector());
  mfem::Vector test_vec(3);
  test_vec(0)                 = 1;
  test_vec(1)                 = 2;
  test_vec(2)                 = 3;
  const auto& func            = coef_opts.scalar_function;
  auto        expected_result = test_vec(1) * 2 + test_vec(2);
  EXPECT_DOUBLE_EQ(func(test_vec, 0.0), expected_result);
  EXPECT_NO_THROW(coef_opts.constructScalar());
}

TEST_F(InputTest, coef_build_constant_scalar)
{
  reader_->parseString("coef_opts = { constant = 2.5 }");
  auto& coef_table = inlet_->addTable("coef_opts");
  input::CoefficientInputOptions::defineInputFileSchema(coef_table);
  auto coef_opts = coef_table.get<input::CoefficientInputOptions>();
  EXPECT_EQ(coef_opts.component, -1);
  EXPECT_FALSE(coef_opts.isVector());
  EXPECT_DOUBLE_EQ(*coef_opts.constant_scalar, 2.5);
  EXPECT_NO_THROW(coef_opts.constructScalar());
}

TEST_F(InputTest, coef_build_piecewise_constant_scalar)
{
  reader_->parseString("coef_opts = { piecewise_constant = { [1] = 2.5, [3] = 3.0 } } ");
  auto& coef_table = inlet_->addTable("coef_opts");
  input::CoefficientInputOptions::defineInputFileSchema(coef_table);
  auto coef_opts = coef_table.get<input::CoefficientInputOptions>();
  EXPECT_EQ(coef_opts.component, -1);
  EXPECT_FALSE(coef_opts.isVector());
  EXPECT_DOUBLE_EQ(coef_opts.pw_const_scalar[1], 2.5);
  EXPECT_DOUBLE_EQ(coef_opts.pw_const_scalar[3], 3.0);
  EXPECT_NO_THROW(coef_opts.constructScalar());
}

TEST_F(InputTest, coef_build_scalar_timedep)
{
  reader_->parseString("coef_opts = { scalar_function = function(v, t) return (v.y * 2 + v.z) * t end, component = 1}");
  auto& coef_table = inlet_->addTable("coef_opts");
  input::CoefficientInputOptions::defineInputFileSchema(coef_table);
  auto coef_opts = coef_table.get<input::CoefficientInputOptions>();
  EXPECT_EQ(coef_opts.component, 1);
  EXPECT_FALSE(coef_opts.isVector());
  mfem::Vector test_vec(3);
  test_vec(0)                  = 1;
  test_vec(1)                  = 2;
  test_vec(2)                  = 3;
  const auto&  func            = coef_opts.scalar_function;
  const double time            = 6.7;
  auto         expected_result = (test_vec(1) * 2 + test_vec(2)) * time;
  EXPECT_DOUBLE_EQ(func(test_vec, time), expected_result);
  EXPECT_NO_THROW(coef_opts.constructScalar());
}

TEST_F(InputTest, coef_build_vec_from_scalar)
{
  reader_->parseString("coef_opts = { scalar_function = function(v) return v.y * 2 + v.z end, component = 1}");
  auto& coef_table = inlet_->addTable("coef_opts");
  input::CoefficientInputOptions::defineInputFileSchema(coef_table);
  auto coef_opts = coef_table.get<input::CoefficientInputOptions>();
  EXPECT_THROW(coef_opts.constructVector(), SlicErrorException);
}

TEST_F(InputTest, coef_build_vector)
{
  reader_->parseString("coef_opts = { vector_function = function(v) return Vector.new(v.y * 2, v.z, v.x) end }");
  auto& coef_table = inlet_->addTable("coef_opts");
  input::CoefficientInputOptions::defineInputFileSchema(coef_table);
  auto coef_opts = coef_table.get<input::CoefficientInputOptions>();
  EXPECT_TRUE(coef_opts.isVector());
  mfem::Vector test_vec(3);
  test_vec(0)       = 1;
  test_vec(1)       = 2;
  test_vec(2)       = 3;
  const auto&  func = coef_opts.vector_function;
  mfem::Vector expected_result(3);
  expected_result(0) = test_vec(1) * 2;
  expected_result(1) = test_vec(2);
  expected_result(2) = test_vec(0);
  mfem::Vector result(3);
  func(test_vec, 0.0, result);
  for (int i = 0; i < result.Size(); i++) {
    EXPECT_DOUBLE_EQ(result[i], expected_result[i]);
  }
  EXPECT_NO_THROW(coef_opts.constructVector());
}

TEST_F(InputTest, coef_build_constant_vector)
{
  reader_->parseString("coef_opts = { constant_vector = { x = 0.0, y = 1.0, z = 2.0 } }");
  auto& coef_table = inlet_->addTable("coef_opts");
  input::CoefficientInputOptions::defineInputFileSchema(coef_table);
  auto coef_opts = coef_table.get<input::CoefficientInputOptions>();
  EXPECT_TRUE(coef_opts.isVector());
  mfem::Vector expected_result(3);
  expected_result(0) = 0.0;
  expected_result(1) = 1.0;
  expected_result(2) = 2.0;

  for (int i = 0; i < coef_opts.constant_vector->Size(); i++) {
    EXPECT_DOUBLE_EQ((*coef_opts.constant_vector)[i], expected_result[i]);
  }
  EXPECT_NO_THROW(coef_opts.constructVector());
}

TEST_F(InputTest, coef_build_piecewise_constant_vector)
{
  reader_->parseString(
      "coef_opts = { piecewise_constant_vector = { [1] = { x = 0.0, y = 1.0 }, [4] = {x = -2.0, y = 1.0} } }");
  auto& coef_table = inlet_->addTable("coef_opts");
  input::CoefficientInputOptions::defineInputFileSchema(coef_table);
  auto coef_opts = coef_table.get<input::CoefficientInputOptions>();
  EXPECT_TRUE(coef_opts.isVector());
  mfem::Vector expected_result_1(2);
  mfem::Vector expected_result_4(2);
  expected_result_1(0) = 0.0;
  expected_result_1(1) = 1.0;
  expected_result_4(0) = -2.0;
  expected_result_4(1) = 1.0;

  for (int i = 0; i < 2; ++i) {
    EXPECT_DOUBLE_EQ(expected_result_1(i), coef_opts.pw_const_vector[1](i));
    EXPECT_DOUBLE_EQ(expected_result_4(i), coef_opts.pw_const_vector[4](i));
  }

  EXPECT_NO_THROW(coef_opts.constructVector());
}

TEST_F(InputTest, coef_build_vector_timedep)
{
  reader_->parseString("coef_opts = { vector_function = function(v, t) return Vector.new(v.y * 2, v.z, v.x) * t end }");
  auto& coef_table = inlet_->addTable("coef_opts");
  input::CoefficientInputOptions::defineInputFileSchema(coef_table);
  auto coef_opts = coef_table.get<input::CoefficientInputOptions>();
  EXPECT_TRUE(coef_opts.isVector());
  mfem::Vector test_vec(3);
  test_vec(0)       = 1;
  test_vec(1)       = 2;
  test_vec(2)       = 3;
  const auto&  func = coef_opts.vector_function;
  const double time = 6.7;
  mfem::Vector expected_result(3);
  expected_result(0) = test_vec(1) * 2;
  expected_result(1) = test_vec(2);
  expected_result(2) = test_vec(0);
  expected_result *= time;
  mfem::Vector result(3);
  func(test_vec, time, result);
  for (int i = 0; i < result.Size(); i++) {
    EXPECT_DOUBLE_EQ(result[i], expected_result[i]);
  }
  EXPECT_NO_THROW(coef_opts.constructVector());
}

TEST_F(InputTest, coef_build_scalar_from_vec)
{
  reader_->parseString("coef_opts = { vector_function = function(v) return Vector.new(v.y * 2, v.z, v.x) end }");
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
