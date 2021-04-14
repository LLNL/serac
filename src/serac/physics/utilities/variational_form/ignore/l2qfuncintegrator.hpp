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
class L2QFunctionIntegrator : public GenericIntegrator {
public:
  L2QFunctionIntegrator(qfunc_type f, Mesh& m)
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
void L2QFunctionIntegrator<qfunc_type>::Apply(const Vector& x, Vector& y) const
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
void L2QFunctionIntegrator<qfunc_type>::Apply2D(const Vector& u_in_, Vector& y_) const
{
  int NE = ne;

  using element_type        = finite_element<::Geometry::Quadrilateral, L2<D1D - 1>>;
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

      auto N = element_type::shape_functions(xi);

      auto u_q = dot(u_local, N);

      tensor<double, 2> x = {X(q, 0, e), X(q, 1, e)};

      auto f0 = qf(x, u_q);

      y_local += N * (f0 * dx);
    }

    for (int i = 0; i < ndof; i++) {
      y(i, e) += y_local[i];
    }
  }
}

template <typename qfunc_type>
void L2QFunctionIntegrator<qfunc_type>::ApplyGradient(const Vector& x, const Vector& v, Vector& y) const
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
void L2QFunctionIntegrator<qfunc_type>::ApplyGradient2D(const Vector& u_in_, const Vector& v_in_, Vector& y_) const
{
  int NE                    = ne;
  using element_type        = finite_element<::Geometry::Quadrilateral, L2<D1D - 1>>;
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

      auto N = element_type::shape_functions(xi);

      auto u_q = dot(u_local, N);
      auto v_q = dot(v_local, N);

      tensor<double, 2> x = {X(q, 0, e), X(q, 1, e)};

      auto f0 = qf(x, make_dual(u_q));

      if constexpr (!std::is_same_v<typename underlying<decltype(f0)>::type, double>) {
        double f00 = f0.gradient;
        y_local += (N * (f00 * v_q)) * dx;
      }
    }

    for (int i = 0; i < ndof; i++) {
      y(i, e) += y_local[i];
    }
  }
}

}  // namespace mfem
