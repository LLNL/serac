#pragma once

#include "mfem.hpp"
#include "mfem/general/forall.hpp"
#include "genericintegrator.hpp"
#include "tensor.hpp"

#include "finite_element.hpp"

template < Geometry g, int Q > 
constexpr auto quadrature_points() {
  auto x = GaussLegendreNodes<Q>();
  if constexpr (g == Geometry::Quadrilateral) {
    tensor < double, Q * Q, 2 > points{};
    int count = 0;
    for (int j = 0; j < Q; j++) {
      for (int i = 0; i < Q; i++) {
        points[count++] = {x[i], x[j]};
      }
    }
    return points;
  }

  if constexpr (g == Geometry::Hexahedron) {
    tensor < double, Q * Q * Q, 3 > points{};
    return points;
  }
}

#if 0
template < Geometry g, PolynomialDegree p, int Q, int components > 
void fem_kernel(const mfem::Vector & values, mfem::Vector & residuals, mfem::Vector & J, mfem::Vector & W, int num_elements) {

  using element_type = finite_element< g, Family::H1, p, components >;

  static constexpr auto quadrature_rule = 

  // for each element
  for (int e = 0; e < num_elements; e++) {

    // load the values associated with that element
    reduced_tensor < double, element_type::ndof, components > element_values{};
    for (int i = 0; i < element_type::ndof; i++) {
      if constexpr (components == 1) {
        element_values[i] = values[element_type::ndof * e + i];
      } else {
        for (int j = 0; j < components; j++) {
          element_values[i] = values[components * (element_type::ndof * e + i) + j];
        }
      }
    }

    // loop over quadrature points
    reduced_tensor < double, element_type::ndof, components > element_residuals{};
    for (int q = 0; q < Q; q++) {

    }

    // store the element residuals 
    for (int i = 0; i < element_type::ndof; i++) {
      if constexpr (components == 1) {
        residuals[element_type::ndof * e + i] = element_residuals[i];
      } else {
        for (int j = 0; j < components; j++) {
          residuals[components * (element_type::ndof * e + i) + j] = element_residuals[i][j];
        }
      }
    }

  }

}
#endif

namespace mfem {
template <typename T>
struct supported_type {
  static constexpr bool value = false;
};

template <>
struct supported_type<ParMesh> {
  static constexpr bool value = true;
};

template <typename qfunc_type, typename qfunc_grad_type, typename... qfunc_args_type>
class QFunctionIntegrator : public GenericIntegrator {
public:
  QFunctionIntegrator(qfunc_type f, qfunc_grad_type f_grad, qfunc_args_type const&... fargs);

  QFunctionIntegrator(qfunc_type f, qfunc_args_type const&... fargs);

  void Setup(const FiniteElementSpace& fes) override;

  void Apply(const Vector&, Vector&) const override;

  // y += F'(x) * v
  void ApplyGradient(const Vector& x, const Vector& v, Vector& y) const override;

protected:
  template <int D1D, int Q1D>
  void Apply2D(const Vector& u_in_, Vector& y_) const;

  template <int D1D, int Q1D>
  void ApplyGradient2D(const Vector& u_in_, const Vector& v_in_, Vector& y_) const;

  auto EvaluateFargValue(const Mesh& m, const double qx, const double qy) const;

  const FiniteElementSpace* fespace;
  const DofToQuad*          maps;  ///< Not owned
  const GeometricFactors*   geom;  ///< Not owned
  int                       dim, ne, nq, dofs1D, quad1D;

  // Geometric factors
  Vector J_;
  Vector W_;

  qfunc_type                     qf;
  qfunc_grad_type                qf_grad;
  std::tuple<qfunc_args_type...> qf_farg_values;
};

template <typename qfunc_type, typename qfunc_grad_type, typename... qfunc_args_type>
QFunctionIntegrator<qfunc_type, qfunc_grad_type, qfunc_args_type...>::QFunctionIntegrator(
    qfunc_type f, qfunc_grad_type df, qfunc_args_type const&... fargs)
    : GenericIntegrator(nullptr), maps(nullptr), geom(nullptr), qf(f), qf_grad(df), qf_farg_values(std::tuple{fargs...})
{
  static_assert((supported_type<qfunc_args_type>::value && ...),
                "Type not supported for parameter expansion. See "
                "documentation for supported types.");
}

template <typename qfunc_type, typename qfunc_grad_type, typename... qfunc_args_type>
QFunctionIntegrator<qfunc_type, qfunc_grad_type, qfunc_args_type...>::QFunctionIntegrator(
    qfunc_type f, qfunc_args_type const&... fargs)
    : GenericIntegrator(nullptr),
      maps(nullptr),
      geom(nullptr),
      qf(f),
      qf_grad(qfunc_grad_type{}),
      qf_farg_values(std::tuple{fargs...})
{
  static_assert((supported_type<qfunc_args_type>::value && ...),
                "Type not supported for parameter expansion. See "
                "documentation for supported types.");
}

template <typename qfunc_type, typename... qfunc_args_type>
QFunctionIntegrator(qfunc_type, qfunc_args_type const&...)
    -> QFunctionIntegrator<qfunc_type, int, qfunc_args_type const&...>;

template <typename qfunc_type, typename qfunc_grad_type, typename... qfunc_args_type>
void QFunctionIntegrator<qfunc_type, qfunc_grad_type, qfunc_args_type...>::Setup(const FiniteElementSpace& fes)
{
  // Assuming the same element type
  fespace    = &fes;
  Mesh* mesh = fes.GetMesh();
  if (mesh->GetNE() == 0) {
    return;
  }
  const FiniteElement& el = *fes.GetFE(0);
  // SERAC EDIT BEGIN
  // ElementTransformation *T = mesh->GetElementTransformation(0);
  // SERAC EDIT END
  const IntegrationRule* ir = nullptr;
  if (!IntRule) {
    IntRule = &IntRules.Get(el.GetGeomType(), el.GetOrder() * 2);
  }
  ir = IntRule;

  dim    = mesh->Dimension();
  ne     = fes.GetMesh()->GetNE();
  nq     = ir->GetNPoints();
  geom   = mesh->GetGeometricFactors(*ir, GeometricFactors::COORDINATES | GeometricFactors::JACOBIANS);
  maps   = &el.GetDofToQuad(*ir, DofToQuad::TENSOR);
  dofs1D = maps->ndof;
  quad1D = maps->nqpt;
  //    pa_data.SetSize(ne * nq, Device::GetDeviceMemoryType());

  W_.SetSize(nq, Device::GetDeviceMemoryType());
  W_.GetMemory().CopyFrom(ir->GetWeights().GetMemory(), nq);

  // J.SetSize(ne * nq, Device::GetDeviceMemoryType());
  J_ = geom->J;
}

template <typename qfunc_type, typename qfunc_grad_type, typename... qfunc_args_type>
auto QFunctionIntegrator<qfunc_type, qfunc_grad_type, qfunc_args_type...>::EvaluateFargValue(const Mesh&  m,
                                                                                             const double qx,
                                                                                             const double qy) const
{
  Vector trip(3);
  trip                      = 0.0;
  ElementTransformation* tr = const_cast<Mesh&>(m).GetElementTransformation(0);
  tr->Transform(IntRule->IntPoint(qx + quad1D * qy), trip);
  return tensor<double, 3>{{trip(0), trip(1), m.SpaceDimension() == 2 ? 0.0 : trip(2)}};
}

template <typename qfunc_type, typename qfunc_grad_type, typename... qfunc_args_type>
void QFunctionIntegrator<qfunc_type, qfunc_grad_type, qfunc_args_type...>::Apply(const Vector& x, Vector& y) const
{
  if (dim == 2) {
    switch ((dofs1D << 4) | quad1D) {
      case 0x22:
        return Apply2D<2, 2>(x, y);
      case 0x33:
        return Apply2D<3, 3>(x, y);
      case 0x44:
        return Apply2D<4, 4>(x, y);
      default:
        MFEM_ASSERT(false, "NOPE");
    }
  }
}

template <typename qfunc_type, typename qfunc_grad_type, typename... qfunc_args_type>
template <int D1D, int Q1D>
void QFunctionIntegrator<qfunc_type, qfunc_grad_type, qfunc_args_type...>::Apply2D(const Vector& u_in_,
                                                                                   Vector&       y_) const
{
  int NE = ne;

  using element_type = finite_element< ::Geometry::Quadrilateral, Family::H1, static_cast< PolynomialDegree >(D1D-1) >;

  static constexpr auto qpts = GaussLegendreNodes<Q1D>(0.0, 1.0);

  auto v1d     = Reshape(maps->B.Read(), Q1D, D1D);
  auto dv1d_dX = Reshape(maps->G.Read(), Q1D, D1D);
  // (NQ x SDIM x DIM x NE)
  auto J = Reshape(J_.Read(), Q1D, Q1D, 2, 2, NE);
  auto W = Reshape(W_.Read(), Q1D, Q1D);
  auto u = Reshape(u_in_.Read(), D1D, D1D, NE);
  auto u2 = Reshape(u_in_.Read(), D1D * D1D, NE);
  auto y = Reshape(y_.ReadWrite(), D1D, D1D, NE);

  // MFEM_FORALL(e, NE, {
  for (int e = 0; e < NE; e++) {
    // loop over quadrature points
    tensor u_local = make_tensor<D1D * D1D>([&u2, e](int i){ return u2(i, e); });

    tensor< double, D1D, D1D > y_local{};
    for (int qy = 0; qy < Q1D; ++qy) {
      for (int qx = 0; qx < Q1D; ++qx) {
        double u_q        = 0.0;
        double du_dX_q[2] = {0.0, 0.0};
        for (int ix = 0; ix < D1D; ix++) {
          for (int iy = 0; iy < D1D; iy++) {
            u_q += u(ix, iy, e) * v1d(qx, ix) * v1d(qy, iy);
            du_dX_q[0] += u(ix, iy, e) * dv1d_dX(qx, ix) * v1d(qy, iy);
            du_dX_q[1] += u(ix, iy, e) * v1d(qx, ix) * dv1d_dX(qy, iy);
          }
        }

        tensor< double, 2> xi{qpts[qx], qpts[qy]};

        std::cout << "1d shape functions: " << std::endl;
        std::cout << GaussLobattoInterpolation01<D1D>(qpts[qx]) << std::endl;
        for (int i = 0; i < D1D; i++) {
          std::cout << v1d(qx, i) << " ";
        }
        std::cout << std::endl;

        std::cout << "1d shape function gradients: " << std::endl;
        std::cout << GaussLobattoInterpolationDerivative01<D1D>(qpts[qx]) << std::endl;
        for (int i = 0; i < D1D; i++) {
          std::cout << dv1d_dX(qx, i) << " ";
        }
        std::cout << std::endl;

        auto U = dot(u_local, element_type::shape_functions(xi));
        auto dU_dX = dot(u_local, element_type::shape_function_gradients(xi));

        std::cout << U << " " << u_q << std::endl;
        std::cout << dU_dX << " " << du_dX_q[0] << " " << du_dX_q[1] << std::endl;
        std::cout << std::endl;

        // du_dx_q = invJ^T * du_dX_q
        //         = (adjJ^T * du_dX_q) / detJ
        double J_q[2][2] = {{J(qx, qy, 0, 0, e), J(qx, qy, 0, 1, e)},   // J_q[0][0], J_q[0][1]
                            {J(qx, qy, 1, 0, e), J(qx, qy, 1, 1, e)}};  // J_q[1][0], J_q[1][1]

        double detJ_q = (J_q[0][0] * J_q[1][1]) - (J_q[0][1] * J_q[1][0]);

        double adjJ[2][2] = {{J_q[1][1], -J_q[0][1]}, {-J_q[1][0], J_q[0][0]}};

        tensor<double, 2> du_dx_q = {(adjJ[0][0] * du_dX_q[0] + adjJ[1][0] * du_dX_q[1]) / detJ_q,
                                     (adjJ[0][1] * du_dX_q[0] + adjJ[1][1] * du_dX_q[1]) / detJ_q};

        auto processed_qf_farg_values = std::apply(
            [=](auto&... a) { return std::make_tuple(u_q, du_dx_q, EvaluateFargValue(a, qx, qy)...); }, qf_farg_values);

        auto [f0, f1] = std::apply(qf, processed_qf_farg_values);

        double f0_X = f0 * detJ_q;

        // f1_X = invJ * f1 * detJ
        //      = adjJ * f1
        double f1_X[2] = {
            adjJ[0][0] * f1[0] + adjJ[0][1] * f1[1],
            adjJ[1][0] * f1[0] + adjJ[1][1] * f1[1],
        };

        for (int ix = 0; ix < D1D; ix++) {
          for (int iy = 0; iy < D1D; iy++) {
            // accumulate v * f0 + dot(dv_dx, f1)
            y_local[ix][iy] += (f0_X * v1d(qx, ix) * v1d(qy, iy) + f1_X[0] * dv1d_dX(qx, ix) * v1d(qy, iy) +
                             f1_X[1] * dv1d_dX(qy, iy) * v1d(qx, ix)) *
                            W(qx, qy);
          }
        }
      }
    }

    for (int ix = 0; ix < D1D; ix++) {
      for (int iy = 0; iy < D1D; iy++) {
        y(ix, iy, e) += y_local[ix][iy];
      }
    }

  }
  // });
}

template <typename qfunc_type, typename qfunc_grad_type, typename... qfunc_args_type>
void QFunctionIntegrator<qfunc_type, qfunc_grad_type, qfunc_args_type...>::ApplyGradient(const Vector& x,
                                                                                         const Vector& v,
                                                                                         Vector&       y) const
{
  ApplyGradient2D<2, 2>(x, v, y);
}

template <typename qfunc_type, typename qfunc_grad_type, typename... qfunc_args_type>
template <int D1D, int Q1D>
void QFunctionIntegrator<qfunc_type, qfunc_grad_type, qfunc_args_type...>::ApplyGradient2D(const Vector& u_in_,
                                                                                           const Vector& v_in_,
                                                                                           Vector&       y_) const
{
  int NE = ne;

  auto v1d     = Reshape(maps->B.Read(), Q1D, D1D);
  auto dv1d_dX = Reshape(maps->G.Read(), Q1D, D1D);
  // (NQ x SDIM x DIM x NE)
  auto J = Reshape(J_.Read(), Q1D, Q1D, 2, 2, NE);
  auto W = Reshape(W_.Read(), Q1D, Q1D);
  auto u = Reshape(u_in_.Read(), D1D, D1D, NE);
  auto v = Reshape(v_in_.Read(), D1D, D1D, NE);
  auto y = Reshape(y_.ReadWrite(), D1D, D1D, NE);

  for (int e = 0; e < NE; e++) {
    // loop over quadrature points
    for (int qy = 0; qy < Q1D; ++qy) {
      for (int qx = 0; qx < Q1D; ++qx) {
        double u_q        = 0.0;
        double du_dX_q[2] = {0.0};
        double v_q        = 0.0;
        double dv_dX_q[2] = {0.0};
        for (int ix = 0; ix < D1D; ix++) {
          for (int iy = 0; iy < D1D; iy++) {
            u_q += u(ix, iy, e) * v1d(qx, ix) * v1d(qy, iy);

            du_dX_q[0] += u(ix, iy, e) * dv1d_dX(qx, ix) * v1d(qy, iy);
            du_dX_q[1] += u(ix, iy, e) * v1d(qx, ix) * dv1d_dX(qy, iy);

            v_q += v(ix, iy, e) * v1d(qx, ix) * v1d(qy, iy);

            dv_dX_q[0] += v(ix, iy, e) * dv1d_dX(qx, ix) * v1d(qy, iy);
            dv_dX_q[1] += v(ix, iy, e) * v1d(qx, ix) * dv1d_dX(qy, iy);
          }
        }

        // du_dx_q = invJ^T * du_dX_q
        //         = (adjJ^T * du_dX_q) / detJ
        double J_q[2][2] = {{J(qx, qy, 0, 0, e), J(qx, qy, 0, 1, e)},   // J_q[0][0], J_q[0][1]
                            {J(qx, qy, 1, 0, e), J(qx, qy, 1, 1, e)}};  // J_q[1][0], J_q[1][1]

        double detJ_q = (J_q[0][0] * J_q[1][1]) - (J_q[0][1] * J_q[1][0]);

        double adjJ[2][2] = {{J_q[1][1], -J_q[0][1]}, {-J_q[1][0], J_q[0][0]}};

        tensor<double, 2> du_dx_q = {(adjJ[0][0] * du_dX_q[0] + adjJ[1][0] * du_dX_q[1]) / detJ_q,
                                     (adjJ[0][1] * du_dX_q[0] + adjJ[1][1] * du_dX_q[1]) / detJ_q};

        double dv_dx_q[2] = {(adjJ[0][0] * dv_dX_q[0] + adjJ[1][0] * dv_dX_q[1]) / detJ_q,
                             (adjJ[0][1] * dv_dX_q[0] + adjJ[1][1] * dv_dX_q[1]) / detJ_q};

        // compute dF(u, du)/du
        auto processed_qf_farg_values_u = std::apply(
            [=](auto&... a) { return std::make_tuple(derivative_wrt(u_q), du_dx_q, EvaluateFargValue(a, qx, qy)...); },
            qf_farg_values);

        auto [f0u, f1u] = std::apply(qf, processed_qf_farg_values_u);

        double f00 = 0.0;
        if constexpr (std::is_same_v<decltype(f0u), double>) {
          f00 = 0.0;
        } else {
          f00 = f0u.gradient;
        }

        tensor<double, 2> f10;
        if constexpr (std::is_same_v<decltype(f1u), tensor<double, 2>>) {
          f10 = {0.0, 0.0};
        } else {
          f10 = {f1u[0].gradient, f1u[1].gradient};
        }

        // compute dF(u, du)/ddu
        auto processed_qf_farg_values_du = std::apply(
            [=](auto&... a) { return std::make_tuple(u_q, derivative_wrt(du_dx_q), EvaluateFargValue(a, qx, qy)...); },
            qf_farg_values);

        auto [f0du, f1du] = std::apply(qf, processed_qf_farg_values_du);

        tensor<double, 2> f01;
        if constexpr (std::is_same_v<decltype(f0du), double>) {
          f01 = {0.0, 0.0};
        } else {
          f01 = f0du.gradient;
        }
        tensor<double, 2, 2> f11 = {
            {{f1du[0].gradient[0], f1du[0].gradient[1]}, {f1du[1].gradient[0], f1du[1].gradient[1]}}};

        double W0 = f00 * v_q + f01[0] * dv_dx_q[0] + f01[1] * dv_dx_q[1];

        double W1[2] = {f10[0] * v_q + f11[0][0] * dv_dx_q[0] + f11[0][1] * dv_dx_q[1],
                        f10[1] * v_q + f11[1][0] * dv_dx_q[0] + +f11[1][1] * dv_dx_q[1]};

        double W0_X = W0 * detJ_q;

        // W1_X = invJ * W1 * detJ
        //      = adjJ * W1
        double W1_X[2] = {
            adjJ[0][0] * W1[0] + adjJ[0][1] * W1[1],
            adjJ[1][0] * W1[0] + adjJ[1][1] * W1[1],
        };

        for (int ix = 0; ix < D1D; ix++) {
          for (int iy = 0; iy < D1D; iy++) {
            // @TODO: proper comment
            y(ix, iy, e) += (W0_X * v1d(qx, ix) * v1d(qy, iy) + W1_X[0] * dv1d_dX(qx, ix) * v1d(qy, iy) +
                             W1_X[1] * dv1d_dX(qy, iy) * v1d(qx, ix)) *
                            W(qx, qy);
          }
        }
      }
    }
  }
}

}  // namespace mfem
