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
class VectorH1QFunctionIntegrator : public GenericIntegrator {
public:
  static constexpr int dim = 2;

  VectorH1QFunctionIntegrator(qfunc_type f, Mesh& m)
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

    if (f_mesh->Dimension() != dim) {
      std::cout << "only 2D is supported right now" << std::endl;
      exit(1);
    }

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
  int                       ne, nq, dofs1D, quad1D;

  // Geometric factors
  Vector J_;
  Vector X_;

  qfunc_type qf;
  Mesh&      mesh;
};

template <typename qfunc_type>
void VectorH1QFunctionIntegrator<qfunc_type>::Apply(const Vector& x, Vector& y) const
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

template <int ndof, int components>
inline auto Load(const mfem::DeviceTensor<3, const double>& u, int e)
{
  return make_tensor<components, ndof>([&u, e](int j, int i) { return u(i, j, e); });
}

template <int ndof, int components>
void Add(const mfem::DeviceTensor<3, double>& r_global, tensor<double, ndof, components> r_local, int e)
{
  for (int i = 0; i < ndof; i++) {
    for (int j = 0; j < components; j++) {
      r_global(i, j, e) += r_local[i][j];
    }
  }
}

template <typename qfunc_type>
template <int D1D, int Q1D>
void VectorH1QFunctionIntegrator<qfunc_type>::Apply2D(const Vector& u_in_, Vector& y_) const
{
  int NE = ne;

  using element_type        = finite_element<::Geometry::Quadrilateral, H1<D1D - 1, dim>>;
  static constexpr int ndof = element_type::ndof;

  static constexpr auto rule = GaussQuadratureRule<::Geometry::Quadrilateral, Q1D>();

  auto X = Reshape(X_.Read(), rule.size(), 2, NE);
  auto J = Reshape(J_.Read(), rule.size(), 2, 2, NE);
  auto u = Reshape(u_in_.Read(), ndof, dim, NE);
  auto y = Reshape(y_.ReadWrite(), ndof, dim, NE);

  for (int e = 0; e < NE; e++) {
    tensor u_local = Load<ndof, dim>(u, e);

    tensor<double, ndof, dim> y_local{};
    for (int q = 0; q < static_cast<int>(rule.size()); q++) {
      auto   xi  = rule.points[q];
      auto   dxi = rule.weights[q];
      auto   J_q = make_tensor<dim, dim>([&](int i, int j) { return J(q, i, j, e); });
      double dx  = det(J_q) * dxi;

      auto N     = element_type::shape_functions(xi);
      auto dN_dx = dot(element_type::shape_function_gradients(xi), inv(J_q));

      auto u_q     = dot(u_local, N);
      auto du_dx_q = dot(u_local, dN_dx);

      tensor<double, 2> x = {X(q, 0, e), X(q, 1, e)};

      auto args = std::tuple{x, u_q, du_dx_q};

      auto [f0, f1] = std::apply(qf, args);

      y_local += (outer(N, f0) + dot(dN_dx, f1)) * dx;
    }

    Add(y, y_local, e);
  }
}

template <typename qfunc_type>
void VectorH1QFunctionIntegrator<qfunc_type>::ApplyGradient(const Vector& x, const Vector& v, Vector& y) const
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
void VectorH1QFunctionIntegrator<qfunc_type>::ApplyGradient2D(const Vector& u_in_, const Vector& v_in_,
                                                              Vector& y_) const
{
  int NE                    = ne;
  using element_type        = finite_element<::Geometry::Quadrilateral, H1<D1D - 1, dim>>;
  static constexpr int dim  = element_type::dim;
  static constexpr int ndof = element_type::ndof;

  static constexpr auto rule = GaussQuadratureRule<::Geometry::Quadrilateral, Q1D>();

  auto X = Reshape(X_.Read(), rule.size(), 2, NE);
  auto J = Reshape(J_.Read(), rule.size(), 2, 2, NE);
  auto u = Reshape(u_in_.Read(), ndof, dim, NE);
  auto v = Reshape(v_in_.Read(), ndof, dim, NE);
  auto y = Reshape(y_.ReadWrite(), ndof, dim, NE);

  for (int e = 0; e < NE; e++) {
    tensor u_local = make_tensor<dim, ndof>([&u, e](int j, int i) { return u(i, j, e); });
    tensor v_local = make_tensor<dim, ndof>([&v, e](int j, int i) { return v(i, j, e); });

    tensor<double, ndof, dim> y_local{};

    for (int q = 0; q < static_cast<int>(rule.size()); q++) {
      auto   xi  = rule.points[q];
      auto   dxi = rule.weights[q];
      auto   J_q = make_tensor<dim, dim>([&](int i, int j) { return J(q, i, j, e); });
      double dx  = det(J_q) * dxi;

      tensor N     = element_type::shape_functions(xi);
      tensor dN_dx = dot(element_type::shape_function_gradients(xi), inv(J_q));

      tensor u_q     = dot(u_local, N);
      tensor du_dx_q = dot(u_local, dN_dx);

      tensor v_q     = dot(v_local, N);
      tensor dv_dx_q = dot(v_local, dN_dx);

      tensor<double, 2> x = {X(q, 0, e), X(q, 1, e)};

      auto args = std::tuple_cat(std::tuple{x}, make_dual(u_q, du_dx_q));

      auto [f0, f1] = std::apply(qf, args);

      if constexpr (!std::is_same_v<typename underlying<decltype(f0)>::type, double>) {
        tensor<double, dim, dim>      f00;
        tensor<double, dim, dim, dim> f01;
        for (int i = 0; i < dim; i++) {
          f00[i] = std::get<0>(f0[i].gradient);
          f01[i] = std::get<1>(f0[i].gradient);
        }
        y_local += outer(N, (dot(f00, v_q) + ddot(f01, dv_dx_q))) * dx;
      }

      if constexpr (!std::is_same_v<typename underlying<decltype(f1)>::type, double>) {
        tensor<double, dim, dim, dim>      f10;
        tensor<double, dim, dim, dim, dim> f11;
        for (int i = 0; i < dim; i++) {
          for (int j = 0; j < dim; j++) {
            f10[i][j] = std::get<0>(f1[i][j].gradient);
            f11[i][j] = std::get<1>(f1[i][j].gradient);
          }
        }
        y_local += dot(dN_dx, dot(f10, v_q) + ddot(f11, dv_dx_q)) * dx;
      }
    }

    Add(y, y_local, e);
  }
}

}  // namespace mfem
