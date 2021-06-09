#pragma once

#include "mfem.hpp"
#include "mfem/linalg/dtensor.hpp"

#include "serac/physics/utilities/functional/integral.hpp"

namespace serac {

namespace detail {
template <typename T, typename... Dims>
SERAC_HOST_DEVICE   inline mfem::DeviceTensor<sizeof...(Dims),T> Reshape(T *ptr, Dims... dims)
   {
      return mfem::DeviceTensor<sizeof...(Dims),T>(ptr, dims...);
   }

} // namespace detail


template <Geometry g, typename test, typename trial, int geometry_dim, int spatial_dim, int Q,
typename derivatives_type, typename lambda>
__global__ void evaluation_kernel_cuda(const mfem::Vector& U, mfem::Vector& R, derivatives_type* derivatives_ptr,
const mfem::Vector& J_, const mfem::Vector& X_, int num_elements, lambda qf)
{
using test_element               = finite_element<g, test>;
using trial_element              = finite_element<g, trial>;
using element_residual_type      = typename trial_element::residual_type;
static constexpr int  test_ndof  = test_element::ndof;
static constexpr int  trial_ndof = trial_element::ndof;
static constexpr auto rule       = GaussQuadratureRule<g, Q>();

// mfem provides this information in 1D arrays, so we reshape it
// into strided multidimensional arrays before using
auto X = detail::Reshape(X_.Read(), rule.size(), spatial_dim, num_elements);
auto J = detail::Reshape(J_.Read(), rule.size(), spatial_dim, geometry_dim, num_elements);
auto u = detail::Reshape<trial>(U.Read(), trial_ndof, num_elements);
auto r = detail::Reshape<test>(R.ReadWrite(), test_ndof, num_elements);

// for each element in the domain
for (int e = 0; e < num_elements; e++) {
// get the DOF values for this particular element
tensor u_elem = detail::Load<trial_element>(u, e);

// this is where we will accumulate the element residual tensor
element_residual_type r_elem{};

// for each quadrature point in the element
for (int q = 0; q < static_cast<int>(rule.size()); q++) {
// get the position of this quadrature point in the parent and physical space,
// and calculate the measure of that point in physical space.
auto   xi  = rule.points[q];
auto   dxi = rule.weights[q];
auto   x_q = make_tensor<spatial_dim>([&](int i) { return X(q, i, e); });  // Physical coords of qpt
auto   J_q = make_tensor<spatial_dim, geometry_dim>([&](int i, int j) { return J(q, i, j, e); });
double dx  = detail::Measure(J_q) * dxi;

// evaluate the value/derivatives needed for the q-function at this quadrature point
auto arg = detail::Preprocess<trial_element>(u_elem, xi, J_q);

// evaluate the user-specified constitutive model
//
// note: make_dual(arg) promotes those arguments to dual number types
// so that qf_output will contain values and derivatives
auto qf_output = qf(x_q, make_dual(arg));

// integrate qf_output against test space shape functions / gradients
// to get element residual contributions
r_elem += detail::Postprocess<test_element>(get_value(qf_output), xi, J_q) * dx;

// here, we store the derivative of the q-function w.r.t. its input arguments
//
// this will be used by other kernels to evaluate gradients / adjoints / directional derivatives
derivatives_ptr[e * int(rule.size()) + q] = get_gradient(qf_output);
}

// once we've finished the element integration loop, write our element residuals
// out to memory, to be later assembled into global residuals by mfem
detail::Add(r, r_elem, e);

}

}

}