// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <chrono>
#include <iostream>

#include "mfem.hpp"

#include "serac/serac_config.hpp"
#include "serac/numerics/expr_template_ops.hpp"

#include "axom/core/utilities/Timer.hpp"

#include "serac/physics/utilities/functional/tensor.hpp"
#include "serac/physics/utilities/functional/domain_integral.hpp"
#include "serac/physics/utilities/functional/quadrature.hpp"
#include "serac/physics/utilities/functional/finite_element.hpp"
#include "serac/physics/utilities/functional/tuple_arithmetic.hpp"

namespace mfem {

// PA Diffusion Assemble 3D kernel
void PADiffusionSetup3D_(const int Q1D, const int coeffDim, const int NE, const Array<double>& w, const Vector& j,
                         const Vector& c, Vector& d)
{
  const bool symmetric = (coeffDim != 9);
  const bool const_c   = c.Size() == 1;
  MFEM_VERIFY(coeffDim < 6 || !const_c, "Constant matrix coefficient not supported");
  const auto W = Reshape(w.Read(), Q1D, Q1D, Q1D);
  const auto J = Reshape(j.Read(), Q1D, Q1D, Q1D, 3, 3, NE);
  const auto C = const_c ? Reshape(c.Read(), 1, 1, 1, 1, 1) : Reshape(c.Read(), coeffDim, Q1D, Q1D, Q1D, NE);
  auto       D = Reshape(d.Write(), Q1D, Q1D, Q1D, symmetric ? 6 : 9, NE);
   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qz,z,Q1D)
            {
               const double J11 = J(qx,qy,qz,0,0,e);
               const double J21 = J(qx,qy,qz,1,0,e);
               const double J31 = J(qx,qy,qz,2,0,e);
               const double J12 = J(qx,qy,qz,0,1,e);
               const double J22 = J(qx,qy,qz,1,1,e);
               const double J32 = J(qx,qy,qz,2,1,e);
               const double J13 = J(qx,qy,qz,0,2,e);
               const double J23 = J(qx,qy,qz,1,2,e);
               const double J33 = J(qx,qy,qz,2,2,e);
               const double detJ = J11 * (J22 * J33 - J32 * J23) -
               /* */               J21 * (J12 * J33 - J32 * J13) +
               /* */               J31 * (J12 * J23 - J22 * J13);
               const double w_detJ = W(qx,qy,qz) / detJ;
               // adj(J)
               const double A11 = (J22 * J33) - (J23 * J32);
               const double A12 = (J32 * J13) - (J12 * J33);
               const double A13 = (J12 * J23) - (J22 * J13);
               const double A21 = (J31 * J23) - (J21 * J33);
               const double A22 = (J11 * J33) - (J13 * J31);
               const double A23 = (J21 * J13) - (J11 * J23);
               const double A31 = (J21 * J32) - (J31 * J22);
               const double A32 = (J31 * J12) - (J11 * J32);
               const double A33 = (J11 * J22) - (J12 * J21);

               if (coeffDim == 6 || coeffDim == 9) // Matrix coefficient version
               {
    // Compute entries of R = MJ^{-T} = M adj(J)^T, without det J.
    const double M11 = C(0, qx, qy, qz, e);
    const double M12 = C(1, qx, qy, qz, e);
    const double M13 = C(2, qx, qy, qz, e);
    const double M21 = (!symmetric) ? C(3, qx, qy, qz, e) : M12;
    const double M22 = (!symmetric) ? C(4, qx, qy, qz, e) : C(3, qx, qy, qz, e);
    const double M23 = (!symmetric) ? C(5, qx, qy, qz, e) : C(4, qx, qy, qz, e);
    const double M31 = (!symmetric) ? C(6, qx, qy, qz, e) : M13;
    const double M32 = (!symmetric) ? C(7, qx, qy, qz, e) : M23;
    const double M33 = (!symmetric) ? C(8, qx, qy, qz, e) : C(5, qx, qy, qz, e);

    const double R11 = M11 * A11 + M12 * A12 + M13 * A13;
    const double R12 = M11 * A21 + M12 * A22 + M13 * A23;
    const double R13 = M11 * A31 + M12 * A32 + M13 * A33;
    const double R21 = M21 * A11 + M22 * A12 + M23 * A13;
    const double R22 = M21 * A21 + M22 * A22 + M23 * A23;
    const double R23 = M21 * A31 + M22 * A32 + M23 * A33;
    const double R31 = M31 * A11 + M32 * A12 + M33 * A13;
    const double R32 = M31 * A21 + M32 * A22 + M33 * A23;
    const double R33 = M31 * A31 + M32 * A32 + M33 * A33;

    // Now set D to J^{-1} R = adj(J) R
    D(qx, qy, qz, 0, e) = w_detJ * (A11 * R11 + A12 * R21 + A13 * R31);  // 1,1
    const double D12    = w_detJ * (A11 * R12 + A12 * R22 + A13 * R32);
    D(qx, qy, qz, 1, e) = D12;                                           // 1,2
    D(qx, qy, qz, 2, e) = w_detJ * (A11 * R13 + A12 * R23 + A13 * R33);  // 1,3

    const double D21 = w_detJ * (A21 * R11 + A22 * R21 + A23 * R31);
    const double D22 = w_detJ * (A21 * R12 + A22 * R22 + A23 * R32);
    const double D23 = w_detJ * (A21 * R13 + A22 * R23 + A23 * R33);

    const double D33 = w_detJ * (A31 * R13 + A32 * R23 + A33 * R33);

    D(qx, qy, qz, 3, e) = symmetric ? D22 : D21;  // 2,2 or 2,1
    D(qx, qy, qz, 4, e) = symmetric ? D23 : D22;  // 2,3 or 2,2
    D(qx, qy, qz, 5, e) = symmetric ? D33 : D23;  // 3,3 or 2,3

    if (!symmetric) {
      D(qx, qy, qz, 6, e) = w_detJ * (A31 * R11 + A32 * R21 + A33 * R31);  // 3,1
      D(qx, qy, qz, 7, e) = w_detJ * (A31 * R12 + A32 * R22 + A33 * R32);  // 3,2
      D(qx, qy, qz, 8, e) = D33;                                           // 3,3
    }
               }
               else  // Vector or scalar coefficient version
               {
    const double C1 = const_c ? C(0, 0, 0, 0, 0) : C(0, qx, qy, qz, e);
    const double C2 = const_c ? C(0, 0, 0, 0, 0) : (coeffDim == 3 ? C(1, qx, qy, qz, e) : C(0, qx, qy, qz, e));
    const double C3 = const_c ? C(0, 0, 0, 0, 0) : (coeffDim == 3 ? C(2, qx, qy, qz, e) : C(0, qx, qy, qz, e));

    // detJ J^{-1} J^{-T} = (1/detJ) adj(J) adj(J)^T
    D(qx, qy, qz, 0, e) = w_detJ * (C1 * A11 * A11 + C2 * A12 * A12 + C3 * A13 * A13);  // 1,1
    D(qx, qy, qz, 1, e) = w_detJ * (C1 * A11 * A21 + C2 * A12 * A22 + C3 * A13 * A23);  // 2,1
    D(qx, qy, qz, 2, e) = w_detJ * (C1 * A11 * A31 + C2 * A12 * A32 + C3 * A13 * A33);  // 3,1
    D(qx, qy, qz, 3, e) = w_detJ * (C1 * A21 * A21 + C2 * A22 * A22 + C3 * A23 * A23);  // 2,2
    D(qx, qy, qz, 4, e) = w_detJ * (C1 * A21 * A31 + C2 * A22 * A32 + C3 * A23 * A33);  // 3,2
    D(qx, qy, qz, 5, e) = w_detJ * (C1 * A31 * A31 + C2 * A32 * A32 + C3 * A33 * A33);  // 3,3
               }
}
}  // namespace mfem
}
});
}

template <int T_D1D = 0, int T_Q1D = 0>
static void PADiffusionApply3D_(const int NE, const bool symmetric, const Array<double>& b, const Array<double>& g,
                                const Array<double>& bt, const Array<double>& gt, const Vector& d_, const Vector& x_,
                                Vector& y_, int d1d = 0, int q1d = 0)
{
  const int D1D = T_D1D ? T_D1D : d1d;
  const int Q1D = T_Q1D ? T_Q1D : q1d;
  MFEM_VERIFY(D1D <= MAX_D1D, "");
  MFEM_VERIFY(Q1D <= MAX_Q1D, "");
  auto B  = Reshape(b.Read(), Q1D, D1D);
  auto G  = Reshape(g.Read(), Q1D, D1D);
  auto Bt = Reshape(bt.Read(), D1D, Q1D);
  auto Gt = Reshape(gt.Read(), D1D, Q1D);
  auto D  = Reshape(d_.Read(), Q1D * Q1D * Q1D, symmetric ? 6 : 9, NE);
  auto X  = Reshape(x_.Read(), D1D, D1D, D1D, NE);
  auto Y  = Reshape(y_.ReadWrite(), D1D, D1D, D1D, NE);
  for (int e = 0; e < NE; e++) {
    constexpr int max_D1D = T_D1D ? T_D1D : MAX_D1D;
    constexpr int max_Q1D = T_Q1D ? T_Q1D : MAX_Q1D;
    double        grad[max_Q1D][max_Q1D][max_Q1D][3];
    for (int qz = 0; qz < Q1D; ++qz) {
      for (int qy = 0; qy < Q1D; ++qy) {
        for (int qx = 0; qx < Q1D; ++qx) {
          grad[qz][qy][qx][0] = 0.0;
          grad[qz][qy][qx][1] = 0.0;
          grad[qz][qy][qx][2] = 0.0;
        }
      }
    }
    for (int dz = 0; dz < D1D; ++dz) {
      double gradXY[max_Q1D][max_Q1D][3];
      for (int qy = 0; qy < Q1D; ++qy) {
        for (int qx = 0; qx < Q1D; ++qx) {
          gradXY[qy][qx][0] = 0.0;
          gradXY[qy][qx][1] = 0.0;
          gradXY[qy][qx][2] = 0.0;
        }
      }
      for (int dy = 0; dy < D1D; ++dy) {
        double gradX[max_Q1D][2];
        for (int qx = 0; qx < Q1D; ++qx) {
          gradX[qx][0] = 0.0;
          gradX[qx][1] = 0.0;
        }
        for (int dx = 0; dx < D1D; ++dx) {
          const double s = X(dx, dy, dz, e);
          for (int qx = 0; qx < Q1D; ++qx) {
            gradX[qx][0] += s * B(qx, dx);
            gradX[qx][1] += s * G(qx, dx);
          }
        }
        for (int qy = 0; qy < Q1D; ++qy) {
          const double wy  = B(qy, dy);
          const double wDy = G(qy, dy);
          for (int qx = 0; qx < Q1D; ++qx) {
            const double wx  = gradX[qx][0];
            const double wDx = gradX[qx][1];
            gradXY[qy][qx][0] += wDx * wy;
            gradXY[qy][qx][1] += wx * wDy;
            gradXY[qy][qx][2] += wx * wy;
          }
        }
      }
      for (int qz = 0; qz < Q1D; ++qz) {
        const double wz  = B(qz, dz);
        const double wDz = G(qz, dz);
        for (int qy = 0; qy < Q1D; ++qy) {
          for (int qx = 0; qx < Q1D; ++qx) {
            grad[qz][qy][qx][0] += gradXY[qy][qx][0] * wz;
            grad[qz][qy][qx][1] += gradXY[qy][qx][1] * wz;
            grad[qz][qy][qx][2] += gradXY[qy][qx][2] * wDz;
          }
        }
      }
    }
    // Calculate Dxyz, xDyz, xyDz in plane
    for (int qz = 0; qz < Q1D; ++qz) {
      for (int qy = 0; qy < Q1D; ++qy) {
        for (int qx = 0; qx < Q1D; ++qx) {
          const int    q      = qx + (qy + qz * Q1D) * Q1D;
          const double O11    = D(q, 0, e);
          const double O12    = D(q, 1, e);
          const double O13    = D(q, 2, e);
          const double O21    = symmetric ? O12 : D(q, 3, e);
          const double O22    = symmetric ? D(q, 3, e) : D(q, 4, e);
          const double O23    = symmetric ? D(q, 4, e) : D(q, 5, e);
          const double O31    = symmetric ? O13 : D(q, 6, e);
          const double O32    = symmetric ? O23 : D(q, 7, e);
          const double O33    = symmetric ? D(q, 5, e) : D(q, 8, e);
          const double gradX  = grad[qz][qy][qx][0];
          const double gradY  = grad[qz][qy][qx][1];
          const double gradZ  = grad[qz][qy][qx][2];
          grad[qz][qy][qx][0] = (O11 * gradX) + (O12 * gradY) + (O13 * gradZ);
          grad[qz][qy][qx][1] = (O21 * gradX) + (O22 * gradY) + (O23 * gradZ);
          grad[qz][qy][qx][2] = (O31 * gradX) + (O32 * gradY) + (O33 * gradZ);
        }
      }
    }
    for (int qz = 0; qz < Q1D; ++qz) {
      double gradXY[max_D1D][max_D1D][3];
      for (int dy = 0; dy < D1D; ++dy) {
        for (int dx = 0; dx < D1D; ++dx) {
          gradXY[dy][dx][0] = 0;
          gradXY[dy][dx][1] = 0;
          gradXY[dy][dx][2] = 0;
        }
      }
      for (int qy = 0; qy < Q1D; ++qy) {
        double gradX[max_D1D][3];
        for (int dx = 0; dx < D1D; ++dx) {
          gradX[dx][0] = 0;
          gradX[dx][1] = 0;
          gradX[dx][2] = 0;
        }
        for (int qx = 0; qx < Q1D; ++qx) {
          const double gX = grad[qz][qy][qx][0];
          const double gY = grad[qz][qy][qx][1];
          const double gZ = grad[qz][qy][qx][2];
          for (int dx = 0; dx < D1D; ++dx) {
            const double wx  = Bt(dx, qx);
            const double wDx = Gt(dx, qx);
            gradX[dx][0] += gX * wDx;
            gradX[dx][1] += gY * wx;
            gradX[dx][2] += gZ * wx;
          }
        }
        for (int dy = 0; dy < D1D; ++dy) {
          const double wy  = Bt(dy, qy);
          const double wDy = Gt(dy, qy);
          for (int dx = 0; dx < D1D; ++dx) {
            gradXY[dy][dx][0] += gradX[dx][0] * wy;
            gradXY[dy][dx][1] += gradX[dx][1] * wDy;
            gradXY[dy][dx][2] += gradX[dx][2] * wy;
          }
        }
      }
      for (int dz = 0; dz < D1D; ++dz) {
        const double wz  = Bt(dz, qz);
        const double wDz = Gt(dz, qz);
        for (int dy = 0; dy < D1D; ++dy) {
          for (int dx = 0; dx < D1D; ++dx) {
            Y(dx, dy, dz, e) += ((gradXY[dy][dx][0] * wz) + (gradXY[dy][dx][1] * wz) + (gradXY[dy][dx][2] * wDz));
          }
        }
      }
    }
  }
}
}

template < ::Geometry g, int P, int Q, typename lambda>
void H1_kernel(const mfem::Vector& U, mfem::Vector& R, const mfem::Vector& J_, int num_elements, lambda&& qf)
{
  using trial                      = H1<P>;
  using test                       = H1<P>;
  using test_element               = finite_element<g, trial>;
  using trial_element              = finite_element<g, test>;
  using element_residual_type      = typename trial_element::residual_type;
  static constexpr int  dim        = dimension_of(g);
  static constexpr int  test_ndof  = test_element::ndof;
  static constexpr int  trial_ndof = trial_element::ndof;
  static constexpr auto rule       = GaussQuadratureRule<g, Q>();

  auto J = mfem::Reshape(J_.Read(), rule.size(), dim, dim, num_elements);
  auto u = detail::Reshape<trial>(U.Read(), trial_ndof, num_elements);
  auto r = detail::Reshape<test>(R.ReadWrite(), test_ndof, num_elements);

  for (int e = 0; e < num_elements; e++) {
    tensor u_elem = detail::Load<trial_element>(u, e);

    element_residual_type r_elem{};

    for (int q = 0; q < static_cast<int>(rule.size()); q++) {
      auto   xi  = rule.points[q];
      auto   dxi = rule.weights[q];
      auto   J_q = make_tensor<dim, dim>([&](int i, int j) { return J(q, i, j, e); });
      double dx  = detail::Measure(J_q) * dxi;

      auto dN    = trial_element::shape_function_gradients(xi);
      auto inv_J = inv(J_q);

      auto grad_u = dot(dot(u_elem, dN), inv_J);

      auto qf_output = qf(grad_u);

      r_elem += dot(dN, dot(inv_J, qf_output)) * dx;
    }

    detail::Add(r, r_elem, e);
  }
}

template < ::Geometry g, int P, int Q, typename lambda>
void H1_kernel_constexpr(const mfem::Vector& U, mfem::Vector& R, const mfem::Vector& J_, int num_elements, lambda&& qf)
{
  using trial                      = H1<P>;
  using test                       = H1<P>;
  using test_element               = finite_element<g, trial>;
  using trial_element              = finite_element<g, test>;
  using element_residual_type      = typename trial_element::residual_type;
  static constexpr int  dim        = dimension_of(g);
  static constexpr int  test_ndof  = test_element::ndof;
  static constexpr int  trial_ndof = trial_element::ndof;
  static constexpr auto rule       = GaussQuadratureRule<g, Q>();

  auto J = mfem::Reshape(J_.Read(), rule.size(), dim, dim, num_elements);
  auto u = detail::Reshape<trial>(U.Read(), trial_ndof, num_elements);
  auto r = detail::Reshape<test>(R.ReadWrite(), test_ndof, num_elements);

  for (int e = 0; e < num_elements; e++) {
    tensor u_elem = detail::Load<trial_element>(u, e);

    element_residual_type r_elem{};

    for_constexpr<rule.size()>([&](auto q) {
      static constexpr auto xi  = rule.points[q];
      static constexpr auto dxi = rule.weights[q];
      auto                  J_q = make_tensor<dim, dim>([&](int i, int j) { return J(q, i, j, e); });
      double                dx  = detail::Measure(J_q) * dxi;

      static constexpr auto dN    = trial_element::shape_function_gradients(xi);
      auto                  inv_J = inv(J_q);

      auto grad_u = dot(dot(u_elem, dN), inv_J);

      auto qf_output = qf(grad_u);

      r_elem += dot(dN, dot(inv_J, qf_output)) * dx;
    });

    detail::Add(r, r_elem, e);
  }
}

int main(int argc, char* argv[])
{
  int         p           = 1;
  int         refinements = 0;
  const char* mesh_file   = SERAC_REPO_DIR "/data/meshes/beam-hex.mesh";

  mfem::OptionsParser args(argc, argv);
  args.AddOption(&p, "-p", "--polynomial_order", "");
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&refinements, "-r", "--ref", "");

  args.Parse();
  if (!args.Good()) {
    exit(1);
  }
  args.PrintOptions(std::cout);

  mfem::Mesh mesh(mesh_file, 1, 1);
  for (int l = 0; l < refinements; l++) {
    mesh.UniformRefinement();
  }

  auto                     fec = mfem::H1_FECollection(p, 3);
  mfem::FiniteElementSpace fespace(&mesh, &fec);

  auto num_elements = mesh.GetNE();

  std::cout << "mesh contains " << num_elements << " elements" << std::endl;

  const mfem::FiniteElement&   el   = *(fespace.GetFE(0));
  const mfem::IntegrationRule& ir   = mfem::IntRules.Get(el.GetGeomType(), el.GetOrder() * 2);
  auto                         geom = mesh.GetGeometricFactors(ir, mfem::GeometricFactors::JACOBIANS);

  auto maps = &el.GetDofToQuad(ir, mfem::DofToQuad::TENSOR);

  auto G = fespace.GetElementRestriction(mfem::ElementDofOrdering::LEXICOGRAPHIC);

  mfem::Vector U_L(fespace.GetTrueVSize(), mfem::Device::GetMemoryType());
  mfem::Vector U_E(G->Height(), mfem::Device::GetMemoryType());

  mfem::Vector J_Q = geom->J;

  mfem::Vector R_E(G->Height(), mfem::Device::GetMemoryType());
  mfem::Vector R_E2(G->Height(), mfem::Device::GetMemoryType());
  mfem::Vector R_E3(G->Height(), mfem::Device::GetMemoryType());

  mfem::Vector R_L(fespace.GetTrueVSize(), mfem::Device::GetMemoryType());

  U_L.Randomize();

  axom::utilities::Timer stopwatch[3];

  stopwatch[0].start();
  G->Mult(U_L, U_E);
  stopwatch[0].stop();
  std::cout << "U_L -> U_E time: " << stopwatch[0].elapsed() << std::endl;

  [[maybe_unused]] static constexpr double k               = 1.0;
  constexpr auto                           diffusion_qfunc = [](auto grad_u) {
    return k * grad_u;  // heat_flux
  };

  bool symmetric = false;
  if (p == 1) {
    stopwatch[0].start();
    mfem::PADiffusionApply3D_<2, 2>(num_elements, symmetric, maps->B, maps->G, maps->Bt, maps->Gt, J_Q, U_E, R_E);
    stopwatch[0].stop();

    stopwatch[1].start();
    H1_kernel<Geometry::Hexahedron, 1, 2>(U_E, R_E2, J_Q, num_elements, diffusion_qfunc);
    stopwatch[1].stop();

    stopwatch[2].start();
    H1_kernel_constexpr<Geometry::Hexahedron, 1, 2>(U_E, R_E3, J_Q, num_elements, diffusion_qfunc);
    stopwatch[2].stop();
  }

  if (p == 2) {
    stopwatch[0].start();
    mfem::PADiffusionApply3D_<3, 3>(num_elements, symmetric, maps->B, maps->G, maps->Bt, maps->Gt, J_Q, U_E, R_E);
    stopwatch[0].stop();
    stopwatch[1].start();
    H1_kernel<Geometry::Hexahedron, 2, 3>(U_E, R_E2, J_Q, num_elements, diffusion_qfunc);
    stopwatch[1].stop();
    stopwatch[2].start();
    H1_kernel_constexpr<Geometry::Hexahedron, 2, 3>(U_E, R_E3, J_Q, num_elements, diffusion_qfunc);
    stopwatch[2].stop();
  }

  if (p == 3) {
    stopwatch[0].start();
    mfem::PADiffusionApply3D_<4, 4>(num_elements, symmetric, maps->B, maps->G, maps->Bt, maps->Gt, J_Q, U_E, R_E);
    stopwatch[0].stop();
    stopwatch[1].start();
    H1_kernel<Geometry::Hexahedron, 3, 4>(U_E, R_E2, J_Q, num_elements, diffusion_qfunc);
    stopwatch[1].stop();
    stopwatch[2].start();
    H1_kernel_constexpr<Geometry::Hexahedron, 3, 4>(U_E, R_E3, J_Q, num_elements, diffusion_qfunc);
    stopwatch[2].stop();
  }

  std::cout << "mfem::PADiffusionApply3D time: " << stopwatch[0].elapsed() << " seconds" << std::endl;
  std::cout << "H1_kernel< diffusion > time: " << stopwatch[1].elapsed() << " seconds" << std::endl;
  std::cout << "H1_kernel_constexpr< diffusion > time: " << stopwatch[2].elapsed() << " seconds" << std::endl;
  std::cout << "||R_1||: " << R_E.Norml2() << std::endl;
  std::cout << "||R_2||: " << R_E2.Norml2() << std::endl;
  std::cout << "||R_3||: " << R_E3.Norml2() << std::endl;
}
