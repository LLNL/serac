// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
// # Author: Jonathan Wong @ LLNL.

#include <gtest/gtest.h>

#include <memory>

#include "axom/slic.hpp"
#include "coefficients/stdfunction_coefficient.hpp"
#include "mfem.hpp"

using namespace mfem;
using namespace std;
using namespace serac;

class StdFunctionCoefficientTest : public ::testing::Test {
protected:
  void SetUp()
  {
    // Set up mesh
    dim_    = 3;
    int nex = 4;
    int ney = 4;
    int nez = 4;

    Mesh mesh(nex, ney, nez, mfem::Element::HEXAHEDRON, true);
    pmesh_  = std::shared_ptr<ParMesh>(new ParMesh(MPI_COMM_WORLD, mesh));
    pfes_   = std::shared_ptr<ParFiniteElementSpace>(new ParFiniteElementSpace(
        pmesh_.get(), new H1_FECollection(1, dim_, BasisType::GaussLobatto), 1, Ordering::byNODES));
    pfes_v_ = std::shared_ptr<ParFiniteElementSpace>(new ParFiniteElementSpace(
        pmesh_.get(), new H1_FECollection(1, dim_, BasisType::GaussLobatto), dim_, Ordering::byNODES));

    pfes_l2_ = std::shared_ptr<ParFiniteElementSpace>(
        new ParFiniteElementSpace(pmesh_.get(), new L2_FECollection(0, dim_), 1, Ordering::byNODES));
  }

  void TearDown() {}

  int                                    dim_;
  std::shared_ptr<ParMesh>               pmesh_;
  std::shared_ptr<ParFiniteElementSpace> pfes_;
  std::shared_ptr<ParFiniteElementSpace> pfes_v_;
  std::shared_ptr<ParFiniteElementSpace> pfes_l2_;
};

TEST_F(StdFunctionCoefficientTest, Xtest)
{
  ParGridFunction u(pfes_v_.get());
  ParGridFunction x(pfes_v_.get());
  pmesh_->GetVertices(x);

  double x_mult = 1.5;
  // Here we stretch the "x-component" of the coordinate by x_mult
  StdFunctionVectorCoefficient x_stretch(3, [x_mult](const mfem::Vector& p, mfem::Vector& u, double /* t */) {
    u = p;
    u[0] *= x_mult;
  });

  u.ProjectCoefficient(x_stretch);
  for (int d = 0; d < pfes_v_->GetNDofs(); d++) {
    for (int v = 0; v < dim_; v++) {
      int vdof_index = pfes_v_->DofToVDof(d, v);
      if (v == 0) {
        EXPECT_NEAR(x[vdof_index] * x_mult, u[vdof_index], 1.e-12);
      } else {
        EXPECT_NEAR(x[vdof_index], u[vdof_index], 1.e-12);
      }
    }
  }
}

/**
In this test we assign the upper-right portion of a cube with a different
attribute on-the-fly

1. Create an indicator function for the corner
2. Create an attribute list that pertains to every element of the mesh. We
utilize the default "digitize" function which assigns any coefficient value > 0
to an integer attribute of 2, and otherwise 1. This can be overridden.
3. Create a coefficient of 1, and then restrict it to attribute 2. (Remember
that them mesh technically only has attribute values of 1.)
4. We then use an AttributeModifierCoefficient and the attribute list which will
modify the attributes on-the-fly.

 */
TEST_F(StdFunctionCoefficientTest, AttributeList)
{
  // create an indicator coefficient that paints the upper-right corner of a
  // cube

  StdFunctionCoefficient corner([](mfem::Vector x, double /* t */) { return (x[0] > 0.75 && x[1] > 0.75) ? 1. : 0.; });

  Array<int> attr_list = serac::makeAttributeList(*pmesh_, corner);

  MFEM_VERIFY(attr_list.Size() > 0 && attr_list.Sum() > 0, "Didn't pick up anything");

  ConstantCoefficient one(1.);
  Array<int>          attr(2);
  attr    = 0;
  attr[1] = 1;
  RestrictedCoefficient        restrict_coefficient(one, attr);
  AttributeModifierCoefficient load_bdr(attr_list, restrict_coefficient);

  ParGridFunction c1(pfes_l2_.get());
  ParGridFunction c2(pfes_l2_.get());

  c1.ProjectCoefficient(corner);
  c2.ProjectCoefficient(load_bdr);

  // Ouput for visualization
  std::unique_ptr<VisItDataCollection> visit = std::unique_ptr<VisItDataCollection>(
      new VisItDataCollection("StdFunctionCoefficient.AttributeModifier", pmesh_.get()));

  visit->RegisterField("c1", &c1);
  visit->RegisterField("c2", &c2);
  visit->Save();

  /**
     Check each element in the mesh to make sure it has the corner should be a
     different attribute
  */

  for (int i = 0; i < c1.Size(); i++) {
    EXPECT_NEAR(c1[i], c2[i], 1.e-12);
  }
}

// This time we just set the attributes to the mesh itself
TEST_F(StdFunctionCoefficientTest, AttributeListSet)
{
  // create an indicator coefficient that paints the upper-right corner of a
  // cube

  StdFunctionCoefficient corner([](mfem::Vector x, double /* t */) { return (x[0] > 0.75 && x[1] > 0.75) ? 1. : 0.; });

  Array<int> attr_list = serac::makeAttributeList(*pmesh_, corner);

  SLIC_WARNING_IF(attr_list.Size() > 0 && attr_list.Sum() > 0, "Didn't pick up anything");

  for (int e = 0; e < pmesh_->GetNE(); e++) {
    pmesh_->GetElement(e)->SetAttribute(attr_list[e]);
  }

  // Ouput for visualization
  std::unique_ptr<VisItDataCollection> visit = std::unique_ptr<VisItDataCollection>(
      new VisItDataCollection("StdFunctionCoefficient.AttributeSet", pmesh_.get()));

  visit->Save();
}

/**
   Create the ess_vdof_list for all dofs that are x = 0
 */
TEST_F(StdFunctionCoefficientTest, EssentialBC)
{
  // Create an indicator function to set all vertices that are x=0
  StdFunctionVectorCoefficient zero_bc(3, [](const Vector& x, Vector& X, double /* t */) {
    X = 0.;
    for (int i = 0; i < 3; i++)
      if (abs(x[i]) < 1.e-13) {
        X[i] = 1.;
      }
  });

  ParGridFunction u_find_ess(pfes_v_.get());
  u_find_ess.ProjectCoefficient(zero_bc);

  Array<int> ess_vdof_list = serac::makeEssList(*pfes_v_, zero_bc);

  Vector u_ess(ess_vdof_list.Size());
  u_ess = 0.;

  // Check and make sure all the vertices found in the mesh that satisfy this
  // criterion are in the attribute list
  for (int v = 0; v < pmesh_->GetNV(); v++) {
    double* coords = pmesh_->GetVertex(v);
    if (coords[0] < 1.e-13) {
      EXPECT_NE(ess_vdof_list.Find(v), -1);
    }
  }
}

/**
   Create the ess_vdof_list for all dofs that are x = 0
 */
TEST_F(StdFunctionCoefficientTest, EssentialBCCube)
{
  // Create an indicator function to set vertex at the origin
  StdFunctionVectorCoefficient origin_bc(3, [](const Vector& x, Vector& X, double /* t */) {
    X = 0.;

    if (abs(x[0]) < 1.e-13 && abs(x[1]) < 1.e-13 && abs(x[2]) < 1.e-13) {
      X = 1.;
    }
  });

  Array<int> ess_origin_bc_list = serac::makeEssList(*pfes_v_, origin_bc);

  // Define bottom indicator list
  StdFunctionVectorCoefficient bottom_bc_z(pfes_v_->GetVDim(), [](const Vector& x, Vector& X, double /* t */) {
    X = 0.;

    if (abs(x[2]) < 1.e-13) {
      X[2] = 1.;
    }
  });

  Array<int> ess_bottom_bc_list = serac::makeEssList(*pfes_v_, bottom_bc_z);

  // Define top indicator list
  StdFunctionVectorCoefficient top_bc_z(pfes_v_->GetVDim(), [](const Vector& x, Vector& X, double /* t */) {
    X = 0.;

    if (abs(x[2] - 1.) < 1.e-13) {
      X[2] = 1.;
    }
  });
  Array<int>                   ess_top_bc_list = serac::makeEssList(*pfes_v_, top_bc_z);

  // Project displacement values z = 0.5*z
  StdFunctionVectorCoefficient vals(pfes_v_->GetVDim(), [](const Vector& x, Vector& disp, double /* t */) {
    disp    = 0.;
    disp[2] = x[2] * 0.5;
  });

  ParGridFunction vals_eval(pfes_v_.get());
  vals_eval.ProjectCoefficient(vals);

  Array<int> ess_vdof_bc_list(ess_bottom_bc_list);
  ess_vdof_bc_list.Append(ess_top_bc_list);
  ess_vdof_bc_list.Append(ess_origin_bc_list);

  // In case a bc_vdof comes up multiple times
  ess_vdof_bc_list.Unique();

  Vector ess_vdof_list_vals(ess_vdof_bc_list.Size());
  vals_eval.GetSubVector(ess_vdof_bc_list, ess_vdof_list_vals);

  Vector nonzero(pfes_v_->GetVSize());
  nonzero = 1.;
  nonzero.SetSubVector(ess_vdof_bc_list, ess_vdof_list_vals);
  ess_vdof_bc_list.Print();
  ess_vdof_list_vals.Print();
  nonzero.Print();
}

TEST_F(StdFunctionCoefficientTest, ScalarValuedTimeDerivatives) {

  auto y_func = [](const mfem::Vector & x, double t){
    return x[0] * sin(2 * M_PI * t);
  };

  auto dy_dt_func = [](const mfem::Vector & x, double t){
    return 2 * M_PI * x[0] * cos(2 * M_PI * t);
  };

  auto d2y_dt2_func = [](const mfem::Vector & x, double t){
    return -4 * M_PI * M_PI * x[0] * sin(2 * M_PI * t);
  };

  StdFunctionCoefficient y(y_func);
  StdFunctionCoefficient dy_dt = d_dt(y);
  StdFunctionCoefficient d2y_dt2 = d2_dt2(y);

  ParGridFunction u(pfes_.get());
  ParGridFunction x(pfes_v_.get());
  pmesh_->GetVertices(x);

  y.SetTime(0.5);
  u.ProjectCoefficient(y);
  mfem::Vector vertex(dim_);
  for (int d = 0; d < pfes_v_->GetNDofs(); d++) {
    for (int v = 0; v < dim_; v++) {
      vertex(v) = x[pfes_v_->DofToVDof(d, v)];
    }
    EXPECT_NEAR(y_func(vertex, 0.5), u[d], 1.e-8);
  }

  dy_dt.SetTime(0.5);
  u.ProjectCoefficient(dy_dt);
  for (int d = 0; d < pfes_v_->GetNDofs(); d++) {
    for (int v = 0; v < dim_; v++) {
      vertex(v) = x[pfes_v_->DofToVDof(d, v)];
    }
    EXPECT_NEAR(dy_dt_func(vertex, 0.5), u[d], 1.e-6);
  }

  d2y_dt2.SetTime(0.5);
  u.ProjectCoefficient(d2y_dt2);
  for (int d = 0; d < pfes_v_->GetNDofs(); d++) {
    for (int v = 0; v < dim_; v++) {
      vertex(v) = x[pfes_v_->DofToVDof(d, v)];
    }
    EXPECT_NEAR(d2y_dt2_func(vertex, 0.5), u[d], 1.e-6);
  }

}

TEST_F(StdFunctionCoefficientTest, VectorValuedTimeDerivatives) {

  auto y_func = [](const mfem::Vector & x, mfem::Vector & y, double t){
    y = x * sin(2 * M_PI * t);
  };

  auto dy_dt_func = [](const mfem::Vector & x, mfem::Vector & y, double t){
    y = 2 * M_PI * x * cos(2 * M_PI * t);
  };

  auto d2y_dt2_func = [](const mfem::Vector & x, mfem::Vector & y, double t){
    y = -4 * M_PI * M_PI * x * sin(2 * M_PI * t);
  };

  StdFunctionVectorCoefficient y(dim_, y_func);
  StdFunctionVectorCoefficient dy_dt = d_dt(y);
  StdFunctionVectorCoefficient d2y_dt2 = d2_dt2(y);

  ParGridFunction u(pfes_v_.get());
  ParGridFunction x(pfes_v_.get());
  pmesh_->GetVertices(x);

  y.SetTime(0.5);
  u.ProjectCoefficient(y);
  mfem::Vector vertex(dim_);
  mfem::Vector expected_value(dim_);
  mfem::Vector actual_value(dim_);
  for (int d = 0; d < pfes_v_->GetNDofs(); d++) {
    for (int v = 0; v < dim_; v++) {
      vertex(v) = x[pfes_v_->DofToVDof(d, v)];
      actual_value(v) = u[pfes_v_->DofToVDof(d, v)];
    }

    y_func(vertex, expected_value, 0.5);

    for (int v = 0; v < dim_; v++) {
      EXPECT_NEAR(expected_value[v], actual_value[v], 1.e-8);
    }
  }

  dy_dt.SetTime(0.5);
  u.ProjectCoefficient(dy_dt);
  for (int d = 0; d < pfes_v_->GetNDofs(); d++) {
    for (int v = 0; v < dim_; v++) {
      vertex(v) = x[pfes_v_->DofToVDof(d, v)];
      actual_value(v) = u[pfes_v_->DofToVDof(d, v)];
    }

    dy_dt_func(vertex, expected_value, 0.5);

    for (int v = 0; v < dim_; v++) {
      EXPECT_NEAR(expected_value[v], actual_value[v], 1.e-6);
    }
  }

  d2y_dt2.SetTime(0.5);
  u.ProjectCoefficient(d2y_dt2);
  for (int d = 0; d < pfes_v_->GetNDofs(); d++) {
    for (int v = 0; v < dim_; v++) {
      vertex(v) = x[pfes_v_->DofToVDof(d, v)];
      actual_value(v) = u[pfes_v_->DofToVDof(d, v)];
    }

    d2y_dt2_func(vertex, expected_value, 0.5);

    for (int v = 0; v < dim_; v++) {
      EXPECT_NEAR(expected_value[v], actual_value[v], 1.e-6);
    }
  }

}

//------------------------------------------------------------------------------
#include "axom/slic/core/UnitTestLogger.hpp"
using axom::slic::UnitTestLogger;

int main(int argc, char* argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  MPI_Init(&argc, &argv);

  UnitTestLogger logger;  // create & initialize test logger, finalized when exiting main scope

  result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}
