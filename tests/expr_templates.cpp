// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "common/expression_templates.hpp"

#if 0
  // Save a copy of the current state vector
  y_ = u;

  // Solve the equation:
  //    du_dt = M^{-1}*[-K(u + dt*du_dt)]
  // for du_dt
  if (dt != old_dt_) {
    T_mat_.reset(mfem::Add(1.0, *M_mat_, dt, *K_mat_));

    // Eliminate the essential DOFs from the T matrix
    for (auto& bc : ess_bdr_) {
      T_e_mat_.reset(T_mat_->EliminateRowsCols(bc.getTrueDofs()));
    }
    T_solver_.SetOperator(*T_mat_);
  }

  // Apply the boundary conditions
  *bc_rhs_ = *rhs_;
  x_       = 0.0;

  for (auto& bc : ess_bdr_) {
    bc.projectBdr(*state_gf_, t);
    state_gf_->SetFromTrueDofs(y_);
    state_gf_->GetTrueDofs(y_);
    bc.eliminateToRHS(*K_mat_, y_, *bc_rhs_);
  }
  K_mat_->Mult(y_, z_);
  z_.Neg();
  z_.Add(1.0, *bc_rhs_);
  T_solver_.Mult(z_, du_dt);
#endif

int main()
{
  constexpr int     m = 8;
  constexpr int     n = 10;
  mfem::Vector      a(m);
  mfem::Vector      b(m);
  mfem::Vector      c(n);
  mfem::DenseMatrix K(m, n);

  for (int i = 0; i < m; i++) {
    a[i] = i;
    b[i] = i * i;
  }

  for (int i = 0; i < m; i++) {
    c[i] = i * i * i;
    for (int j = 0; j < n; j++) {
      K(i, j) = 2 * (i == j) - (i == (j + 1)) - (i == (j - 1));
    }
  }

  auto f = [](const auto& a, const auto& b, const auto& Kc) { return -a + b * 3.0 - 0.3 * Kc; };

  auto result = evaluate(-a + b * 3.0 - 0.3 * (K * c));

  mfem::Vector Kc(m);
  K.Mult(c, Kc);

  auto result2 = evaluate(f(a, b, Kc));
  for (int i = 0; i < m; i++) {
    std::cout << result[i] << ' ' << result2[i] << ' ' << f(a[i], b[i], Kc[i]) << std::endl;
  }

  for (int i = 0; i < m; i++) {
    std::cout << result[i] << ' ' << f(a[i], b[i], Kc[i]) << std::endl;
  }
}
