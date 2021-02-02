#pragma once

#include "mfem.hpp"
#include "mfem/general/forall.hpp"
#include "genericintegrator.hpp"
#include "tensor.hpp"

#include "finite_element.hpp"

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

  static constexpr auto rule = GaussQuadratureRule<::Geometry::Quadrilateral, Q1D>();

  // (NQ x SDIM x DIM x NE)
  auto J = Reshape(J_.Read(), Q1D, Q1D, 2, 2, NE);
  //auto W = Reshape(W_.Read(), Q1D, Q1D);
  //auto u = Reshape(u_in_.Read(), D1D, D1D, NE);
  auto u = Reshape(u_in_.Read(), D1D * D1D, NE);
  auto y = Reshape(y_.ReadWrite(), D1D * D1D, NE);

  // MFEM_FORALL(e, NE, {
  for (int e = 0; e < NE; e++) {
    tensor u_local = make_tensor<D1D * D1D>([&u, e](int i){ return u(i, e); });

    tensor <double, D1D * D1D > y_local{};
    for (int qy = 0; qy < Q1D; ++qy) {
      for (int qx = 0; qx < Q1D; ++qx) {

        int q = qx + Q1D * qy;

        auto xi = rule.points[q];
        auto dxi = rule.weights[q];

        auto N = element_type::shape_functions(xi);
        auto dN_dxi = element_type::shape_function_gradients(xi);

        auto u_q = dot(u_local, N);
        auto du_dX_q = dot(u_local, dN_dxi);

        auto J_q = make_tensor< dim, dim >([&](int i, int j){ return J(qx, qy, i, j, e); });

        double dx = det(J_q) * dxi;

        auto du_dx_q = dot(du_dX_q, inv(J_q));

        auto args = std::tuple{IntegrationPointPosition(qx + Q1D * qy, e), u_q, du_dx_q};

        auto [f0, f1] = std::apply(qf, args);

        // chain rule: dN_dx = dN_dxi * dxi_dx = dN_dxi * inv(dx_dxi)
        // ===>        dN_dx * f1 = dN_dxi * inv(dx_dxi) * f1
        // we perform (inv(dx_dxi) * f1) first, because f1 has smaller
        // dimensions than dN_dxi, so it should be less expensive
        y_local += (N * f0 + dot(dN_dxi, dot(inv(J_q), f1))) * dx;

      }
    }

    for (int i = 0; i < D1D * D1D; i++) {
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

  static constexpr auto rule = GaussQuadratureRule<::Geometry::Quadrilateral, Q1D>();

  //auto v1d     = Reshape(maps->B.Read(), Q1D, D1D);
  //auto dv1d_dX = Reshape(maps->G.Read(), Q1D, D1D);
  // (NQ x SDIM x DIM x NE)
  auto J = Reshape(J_.Read(), Q1D, Q1D, 2, 2, NE);
  auto u = Reshape(u_in_.Read(), D1D * D1D, NE);
  auto v = Reshape(v_in_.Read(), D1D * D1D, NE);
  auto y = Reshape(y_.ReadWrite(), D1D * D1D, NE);

  for (int e = 0; e < NE; e++) {
    tensor u_local = make_tensor<D1D * D1D>([&u, e](int i){ return u(i, e); });
    tensor v_local = make_tensor<D1D * D1D>([&v, e](int i){ return v(i, e); });

    tensor< double, D1D * D1D > y_local{};
    for (int qy = 0; qy < Q1D; ++qy) {
      for (int qx = 0; qx < Q1D; ++qx) {

        int q = qx + Q1D * qy;

        auto xi = rule.points[q];
        auto dxi = rule.weights[q];

        auto N = element_type::shape_functions(xi);
        auto dN_dxi = element_type::shape_function_gradients(xi);

        auto u_q = dot(u_local, N);
        auto du_dX_q = dot(u_local, dN_dxi);

        auto v_q = dot(v_local, N);
        auto dv_dX_q = dot(v_local, dN_dxi);

        auto J_q = make_tensor< dim, dim >([&](int i, int j){ return J(qx, qy, i, j, e); });

        double dx = det(J_q) * dxi;

        auto du_dx_q = dot(du_dX_q, inv(J_q));
        auto dv_dx_q = dot(dv_dX_q, inv(J_q));

        // compute dF(u, du)/du
        auto args0 = std::tuple{IntegrationPointPosition(qx + Q1D * qy, e), derivative_wrt(u_q), du_dx_q};
        auto [f0u, f1u] = std::apply(qf, args0);

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

        auto args1 = std::tuple{IntegrationPointPosition(qx + Q1D * qy, e), u_q, derivative_wrt(du_dx_q)};
        auto [f0du, f1du] = std::apply(qf, args1);

        tensor<double, 2> f01;
        if constexpr (std::is_same_v<decltype(f0du), double>) {
          f01 = {0.0, 0.0};
        } else {
          f01 = f0du.gradient;
        }
        tensor<double, 2, 2> f11 = {
            {{f1du[0].gradient[0], f1du[0].gradient[1]}, {f1du[1].gradient[0], f1du[1].gradient[1]}}};

        double W0 = f00 * v_q + dot(f01, dv_dx_q);
        tensor W1 = outer(f10, v_q) + dot(f11, dv_dx_q);

        y_local += (N * W0 + dot(dN_dxi, dot(inv(J_q), W1))) * dx;
      }
    }

    for (int i = 0; i < D1D * D1D; i++) {
      y(i, e) += y_local[i];
    }
  }
}

}  // namespace mfem
