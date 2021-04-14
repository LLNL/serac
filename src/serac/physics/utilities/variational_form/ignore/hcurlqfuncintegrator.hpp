#pragma once

#include "mfem.hpp"
#include "mfem/general/forall.hpp"
#include "genericintegrator.hpp"
#include "tensor.hpp"
#include "tuple_arithmetic.hpp"
#include "quadrature.hpp"

#include "finite_element.hpp"

namespace mfem {
template <typename qfunc_type>
class HCurlQFunctionIntegrator : public GenericIntegrator {
public:
  HCurlQFunctionIntegrator(qfunc_type f, Mesh& m)
      : GenericIntegrator(nullptr), maps(nullptr), geom(nullptr), qf(f), mesh(m)
  {
  }

  void Setup(const FiniteElementSpace& fes) override
  {
    // Assuming the same element type
    fespace      = &fes;
    Mesh* f_mesh = fes.GetMesh();
    if (f_mesh->GetNE() == 0) {
      return;
    }
    const FiniteElement& el = *fes.GetFE(0);

    const IntegrationRule* ir = nullptr;
    if (!IntRule) {
      IntRule = &IntRules.Get(el.GetGeomType(), el.GetOrder() * 2);
    }
    ir = IntRule;

    dim    = f_mesh->Dimension();
    ne     = fes.GetMesh()->GetNE();
    nq     = ir->GetNPoints();
    geom   = f_mesh->GetGeometricFactors(*ir, GeometricFactors::COORDINATES | GeometricFactors::JACOBIANS);
    maps   = &el.GetDofToQuad(*ir, DofToQuad::TENSOR);
    dofs1D = maps->ndof;
    quad1D = maps->nqpt;
    //    pa_data.SetSize(ne * nq, Device::GetDeviceMemoryType());

    J_ = geom->J;
    X_ = geom->X;
  };

  void Apply(const Vector&, Vector&) const override;

  // y += F'(x) * v
  void ApplyGradient(const Vector& x, const Vector& v, Vector& y) const override;

protected:
  template <int D1D, int Q1D>
  void Apply2D(const Vector& u_in_, Vector& y_) const;

  template <int D1D, int Q1D>
  void ApplyGradient2D(const Vector& u_in_, const Vector& v_in_, Vector& y_) const;

  const FiniteElementSpace* fespace;
  const DofToQuad*          maps;  ///< Not owned
  const GeometricFactors*   geom;  ///< Not owned
  int                       dim, ne, nq, dofs1D, quad1D;

  // Geometric factors
  Vector J_;
  Vector X_;

  qfunc_type qf;
  Mesh&      mesh;
};

template <typename qfunc_type>
void HCurlQFunctionIntegrator<qfunc_type>::Apply(const Vector& x, Vector& y) const
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

template <typename qfunc_type>
template <int D1D, int Q1D>
void HCurlQFunctionIntegrator<qfunc_type>::Apply2D(const Vector& u_in_, Vector& y_) const
{
  int NE = ne;

  using element_type        = finite_element<::Geometry::Quadrilateral, Hcurl<D1D - 1>>;
  static constexpr int dim  = element_type::dim;
  static constexpr int ndof = element_type::ndof;

  static constexpr auto rule = GaussQuadratureRule<::Geometry::Quadrilateral, Q1D>();

  auto X = Reshape(X_.Read(), rule.size(), 2, NE);
  auto J = Reshape(J_.Read(), rule.size(), 2, 2, NE);
  auto u = Reshape(u_in_.Read(), ndof, NE);
  auto y = Reshape(y_.ReadWrite(), ndof, NE);

  // MFEM_FORALL(e, NE, {
  for (int e = 0; e < NE; e++) {
    tensor u_local = make_tensor<ndof>([&u, e](int i) { return u(i, e); });

    tensor<double, ndof> y_local{};
    for (int q = 0; q < static_cast<int>(rule.size()); q++) {
      auto   xi  = rule.points[q];
      auto   dxi = rule.weights[q];
      auto   J_q = make_tensor<dim, dim>([&](int i, int j) { return J(q, i, j, e); });
      double dx  = det(J_q) * dxi;

      auto N      = dot(element_type::shape_functions(xi), inv(J_q));
      auto curl_N = element_type::shape_function_curl(xi) / det(J_q);

      auto u_q      = dot(u_local, N);
      auto curl_u_q = dot(u_local, curl_N);

      tensor<double, 2> x = {X(q, 0, e), X(q, 1, e)};

      auto args = std::tuple{x, u_q, curl_u_q};

      auto [f0, f1] = std::apply(qf, args);

      y_local += (N * f0 + curl_N * f1) * dx;
    }

    for (int i = 0; i < ndof; i++) {
      y(i, e) += y_local[i];
    }
  }
}

template <typename qfunc_type>
void HCurlQFunctionIntegrator<qfunc_type>::ApplyGradient(const Vector& x, const Vector& v, Vector& y) const
{
  if (dim == 2) {
    switch ((dofs1D << 4) | quad1D) {
      case 0x22:
        return ApplyGradient2D<2, 2>(x, v, y);
      case 0x33:
        return ApplyGradient2D<3, 3>(x, v, y);
      case 0x44:
        return ApplyGradient2D<4, 4>(x, v, y);
      default:
        MFEM_ASSERT(false, "NOPE");
    }
  }
}

template <typename qfunc_type>
template <int D1D, int Q1D>
void HCurlQFunctionIntegrator<qfunc_type>::ApplyGradient2D(const Vector& u_in_, const Vector& v_in_, Vector& y_) const
{
  int NE                    = ne;
  using element_type        = finite_element<::Geometry::Quadrilateral, Hcurl<D1D - 1>>;
  static constexpr int dim  = element_type::dim;
  static constexpr int ndof = element_type::ndof;

  static constexpr auto rule = GaussQuadratureRule<::Geometry::Quadrilateral, Q1D>();

  auto X = Reshape(X_.Read(), rule.size(), 2, NE);
  auto J = Reshape(J_.Read(), rule.size(), 2, 2, NE);
  auto u = Reshape(u_in_.Read(), ndof, NE);
  auto v = Reshape(v_in_.Read(), ndof, NE);
  auto y = Reshape(y_.ReadWrite(), ndof, NE);

  for (int e = 0; e < NE; e++) {
    tensor u_local = make_tensor<ndof>([&u, e](int i) { return u(i, e); });
    tensor v_local = make_tensor<ndof>([&v, e](int i) { return v(i, e); });

    tensor<double, ndof> y_local{};

    for (int q = 0; q < static_cast<int>(rule.size()); q++) {
      auto   xi  = rule.points[q];
      auto   dxi = rule.weights[q];
      auto   J_q = make_tensor<dim, dim>([&](int i, int j) { return J(q, i, j, e); });
      double dx  = det(J_q) * dxi;

      auto N      = dot(element_type::shape_functions(xi), inv(J_q));
      auto curl_N = element_type::shape_function_curl(xi) / det(J_q);

      auto u_q      = dot(u_local, N);
      auto curl_u_q = dot(u_local, curl_N);

      auto v_q      = dot(v_local, N);
      auto curl_v_q = dot(v_local, curl_N);

      tensor<double, 2> x = {X(q, 0, e), X(q, 1, e)};

      auto args = std::tuple_cat(std::tuple{x}, make_dual(u_q, curl_u_q));

      auto [f0, f1] = std::apply(qf, args);

      // the following conditional blocks are to catch the cases where f0 or f1 do not actually
      // depend on the arguments to the q-function.
      //
      // In that case, the dual number types will not propagate through to the return statement,
      // so the output will be a double or a tensor of doubles, rather than dual < ... >
      // or tensor< dual < ... >, n ... >.
      //
      // underlying< ... >::type lets us do some metaprogramming to detect this, and
      // issue a no-op in the event that f0 or f1 does not depend on the input arguments
      //
      // in summary: if the user gives us a function where some of the outputs do not depend on
      // inputs, we can detect this at compile time and skip unnecessary calculation/storage
      if constexpr (!std::is_same_v<typename underlying<decltype(f0)>::type, double>) {
        tensor<double, 2, 2> f00{{{std::get<0>(f0[0].gradient)[0], std::get<0>(f0[0].gradient)[1]},
                                  {std::get<0>(f0[1].gradient)[0], std::get<0>(f0[1].gradient)[1]}}};
        tensor<double, 2>    f01 = {std::get<1>(f0[0].gradient), std::get<1>(f0[1].gradient)};
        y_local += (N * (dot(f00, v_q) + f01 * curl_v_q)) * dx;
      }

      if constexpr (!std::is_same_v<typename underlying<decltype(f1)>::type, double>) {
        tensor<double, 2> f10 = std::get<0>(f1.gradient);
        double            f11 = std::get<1>(f1.gradient);
        y_local += curl_N * (dot(f10, v_q) + f11 * curl_v_q) * dx;
      }
    }

    for (int i = 0; i < ndof; i++) {
      y(i, e) += y_local[i];
    }
  }
}

}  // namespace mfem
