#pragma once

#include "mfem.hpp"
#include "mfem/general/forall.hpp"
#include "genericintegrator.hpp"
#include "tensor.hpp"
#include "tuple_arithmetic.hpp"

#include "finite_element.hpp"

template < typename T >
struct underlying{
  using type = void;
};

template < typename T, int ... n >
struct underlying < tensor < T, n ... > >{
  using type = T;
};

template <>
struct underlying < double >{
  using type = double;
};

template <int n, int dim>
struct QuadratureRule {
  array<double, n>              weights;
  array<tensor<double, dim>, n> points;
  constexpr size_t              size() const { return n; }
};

template <::Geometry g, int Q>
constexpr auto GaussQuadratureRule()
{
  auto x = GaussLegendreNodes<Q>(0.0, 1.0);
  auto w = GaussLegendreWeights<Q>();
  if constexpr (g == Geometry::Quadrilateral) {
    QuadratureRule<Q * Q, 2> rule{};
    int                      count = 0;
    for (int j = 0; j < Q; j++) {
      for (int i = 0; i < Q; i++) {
        rule.points[count]    = {x[i], x[j]};
        rule.weights[count++] = w[i] * w[j];
      }
    }
    return rule;
  }

  if constexpr (g == Geometry::Hexahedron) {
    QuadratureRule<Q * Q * Q, 3> rule{};
    int                          count = 0;
    for (int k = 0; k < Q; k++) {
      for (int j = 0; j < Q; j++) {
        for (int i = 0; i < Q; i++) {
          rule.points[count]    = {x[i], x[j], x[k]};
          rule.weights[count++] = w[i] * w[j] * w[k];
        }
      }
    }
    return rule;
  }
}

namespace mfem {
template <typename T>
struct supported_type {
  static constexpr bool value = false;
};

template <>
struct supported_type<ParMesh> {
  static constexpr bool value = true;
};

template <typename qfunc_type>
class QFunctionIntegrator : public GenericIntegrator {
public:
  QFunctionIntegrator(qfunc_type f, Mesh & m) : GenericIntegrator(nullptr), maps(nullptr), geom(nullptr), qf(f), mesh(m) {}

  void Setup(const FiniteElementSpace& fes) override {
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
  };

  void Apply(const Vector&, Vector&) const override;

  // y += F'(x) * v
  void ApplyGradient(const Vector& x, const Vector& v, Vector& y) const override;

protected:
  template <int D1D, int Q1D>
  void Apply2D(const Vector& u_in_, Vector& y_) const;

  template <int D1D, int Q1D>
  void ApplyGradient2D(const Vector& u_in_, const Vector& v_in_, Vector& y_) const;

  auto IntegrationPointPosition(const int q, const int e) const;

  const FiniteElementSpace* fespace;
  const DofToQuad*          maps;  ///< Not owned
  const GeometricFactors*   geom;  ///< Not owned
  int                       dim, ne, nq, dofs1D, quad1D;

  // Geometric factors
  Vector J_;
  Vector W_;

  qfunc_type                     qf;
  Mesh & mesh;
};

template <typename qfunc_type>
auto QFunctionIntegrator<qfunc_type>::IntegrationPointPosition(const int q, const int e) const
{
  Vector trip(3);
  trip                      = 0.0;
  ElementTransformation* tr = const_cast<Mesh&>(mesh).GetElementTransformation(e);
  tr->Transform(IntRule->IntPoint(q), trip);
  return tensor<double, 3>{{trip(0), trip(1), mesh.SpaceDimension() == 2 ? 0.0 : trip(2)}};
}

template <typename qfunc_type>
void QFunctionIntegrator<qfunc_type>::Apply(const Vector& x, Vector& y) const
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
void QFunctionIntegrator<qfunc_type>::Apply2D(const Vector& u_in_, Vector& y_) const
{
  int NE = ne;

  using element_type = finite_element<::Geometry::Quadrilateral, Family::H1, static_cast<PolynomialDegree>(D1D - 1)>;
  static constexpr int dim = element_type::dim;
  static constexpr int ndof = element_type::ndof;

  static constexpr auto rule = GaussQuadratureRule<::Geometry::Quadrilateral, Q1D>();

  auto J = Reshape(J_.Read(), rule.size(), 2, 2, NE);
  auto u = Reshape(u_in_.Read(), ndof, NE);
  auto y = Reshape(y_.ReadWrite(), ndof, NE);

  // MFEM_FORALL(e, NE, {
  for (int e = 0; e < NE; e++) {
    tensor u_local = make_tensor<ndof>([&u, e](int i){ return u(i, e); });

    tensor <double, ndof > y_local{};
    for (size_t q = 0; q < rule.size(); q++) {
      auto xi = rule.points[q];
      auto dxi = rule.weights[q];
      auto J_q = make_tensor< dim, dim >([&](int i, int j){ return J(q, i, j, e); });
      double dx = det(J_q) * dxi;

      auto N = element_type::shape_functions(xi);
      auto dN_dxi = element_type::shape_function_gradients(xi);

      auto u_q = dot(u_local, N);
      auto du_dx_q = dot(dot(u_local, dN_dxi), inv(J_q));

      auto args = std::tuple{IntegrationPointPosition(q, e), u_q, du_dx_q};

      auto [f0, f1] = std::apply(qf, args);

      // chain rule: dN_dx = dN_dxi * dxi_dx = dN_dxi * inv(dx_dxi)
      // ===>        dN_dx * f1 = dN_dxi * inv(dx_dxi) * f1
      // we perform (inv(dx_dxi) * f1) first, because f1 has smaller
      // dimensions than dN_dxi, so it should be less expensive
      y_local += (N * f0 + dot(dN_dxi, dot(inv(J_q), f1))) * dx;

    }

    for (int i = 0; i < ndof; i++) {
      y(i, e) += y_local[i];
    }

  }
}


template <typename qfunc_type>
void QFunctionIntegrator<qfunc_type>::ApplyGradient(const Vector& x, const Vector& v, Vector& y) const
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
void QFunctionIntegrator<qfunc_type>::ApplyGradient2D(const Vector& u_in_, const Vector& v_in_, Vector& y_) const
{
  int NE             = ne;
  using element_type = finite_element<::Geometry::Quadrilateral, Family::H1, static_cast<PolynomialDegree>(D1D - 1)>;
  static constexpr int dim = element_type::dim;
  static constexpr int ndof = element_type::ndof;

  static constexpr auto rule = GaussQuadratureRule<::Geometry::Quadrilateral, Q1D>();

  auto J = Reshape(J_.Read(), rule.size(), 2, 2, NE);
  auto u = Reshape(u_in_.Read(), ndof, NE);
  auto v = Reshape(v_in_.Read(), ndof, NE);
  auto y = Reshape(y_.ReadWrite(), ndof, NE);

  for (int e = 0; e < NE; e++) {
    tensor u_local = make_tensor<ndof>([&u, e](int i){ return u(i, e); });
    tensor v_local = make_tensor<ndof>([&v, e](int i){ return v(i, e); });

    tensor< double, ndof > y_local{};

    for (size_t q = 0; q < rule.size(); q++) {
      auto xi = rule.points[q];
      auto dxi = rule.weights[q];
      auto J_q = make_tensor< dim, dim >([&](int i, int j){ return J(q, i, j, e); });
      double dx = det(J_q) * dxi;

      auto N = element_type::shape_functions(xi);
      auto dN_dxi = element_type::shape_function_gradients(xi);

      auto u_q = dot(u_local, N);
      auto du_dx_q = dot(dot(u_local, dN_dxi), inv(J_q));

      auto v_q = dot(v_local, N);
      auto dv_dx_q = dot(dot(v_local, dN_dxi), inv(J_q));

      auto x = IntegrationPointPosition(q, e);

      auto args = std::tuple_cat(std::tuple{x}, make_dual(u_q, du_dx_q));

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
        double f00 = std::get<0>(f0.gradient);
        tensor<double, 2> f01 = std::get<1>(f0.gradient);
        y_local += (N * (f00 * v_q + dot(f01, dv_dx_q))) * dx;
      }

      if constexpr (!std::is_same_v<typename underlying<decltype(f1)>::type, double>) {
        tensor<double, 2> f10 = {std::get<0>(f1[0].gradient), std::get<0>(f1[1].gradient)};
        tensor<double, 2, 2> f11{{
          {std::get<1>(f1[0].gradient)[0], std::get<1>(f1[0].gradient)[1]}, 
          {std::get<1>(f1[1].gradient)[0], std::get<1>(f1[1].gradient)[1]}
        }};
        y_local += dot(dN_dxi, dot(inv(J_q), outer(f10, v_q) + dot(f11, dv_dx_q))) * dx;
      }
      
    }

    for (int i = 0; i < ndof; i++) {
      y(i, e) += y_local[i];
    }
  }
}

}  // namespace mfem
