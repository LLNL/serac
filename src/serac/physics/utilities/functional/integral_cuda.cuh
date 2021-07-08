#pragma once

#include "mfem.hpp"
#include "mfem/linalg/dtensor.hpp"

#include "serac/physics/utilities/functional/integral_utilities.hpp"

namespace serac {

namespace detail {
template <typename T, typename... Dims>
__host__ inline mfem::DeviceTensor<sizeof...(Dims),T> Reshape(T *ptr, Dims... dims)
   {
      return mfem::DeviceTensor<sizeof...(Dims),T>(ptr, dims...);
   }


    /// Print cuda Errors
  template <typename S>
  inline void displayLastCUDAErrorMessage(std::ostream & o, const S & prefix) {
      auto error = cudaGetLastError();
      if(error != cudaError::cudaSuccess) {
	o << "Last Cuda Error Message :" << cudaGetErrorString(error) << std::endl;
      } else {
	o << prefix << std::endl;
      }
    }

  inline void displayLastCUDAErrorMessage(std::ostream & o) {
    displayLastCUDAErrorMessage(o, "");
  }


} // namespace detail

  template <Geometry g, typename test, typename trial, int geometry_dim, int spatial_dim, int Q , typename derivatives_type, typename lambda, typename u_elem_type, typename element_residual_type>
  SERAC_HOST_DEVICE element_residual_type eval_quadrature(int e, int q, u_elem_type u_elem, element_residual_type r_elem, mfem::DeviceTensor<2, double> r, derivatives_type * derivatives_ptr, const mfem::DeviceTensor<4, const double> J, const mfem::DeviceTensor<3, const double> X, int num_elements, lambda qf) 
  {
    using test_element               = finite_element<g, test>;
    using trial_element              = finite_element<g, trial>;
    //    using element_residual_type      = typename trial_element::residual_type;
    static constexpr auto rule       = GaussQuadratureRule<g, Q>();

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

    // Note: This pattern appears to result in non-coalesced access
    detail::AccessDerivatives(derivatives_ptr, e , q, rule, num_elements) = get_gradient(qf_output);

    return r_elem;
  }


  template <Geometry g, typename test, typename trial, int geometry_dim, int spatial_dim, int Q, typename derivatives_type, typename lambda>
  __global__ void eval_cuda_element(const mfem::DeviceTensor<2, const double> u, mfem::DeviceTensor<2, double> r, derivatives_type * derivatives_ptr, const mfem::DeviceTensor<4, const double> J, const mfem::DeviceTensor<3, const double> X, int num_elements, lambda qf) {

    using test_element               = finite_element<g, test>;
    using trial_element              = finite_element<g, trial>;
    using element_residual_type      = typename trial_element::residual_type;
    static constexpr auto rule       = GaussQuadratureRule<g, Q>();

    int e = blockIdx.x * blockDim.x + threadIdx.x;
    // for each element in the domain
    if (e < num_elements) {
      // get the DOF values for this particular element
      auto u_elem = detail::Load<trial_element>(u, e);

      // this is where we will accumulate the element residual tensor
      element_residual_type r_elem{};

      // for each quadrature point in the element
      for (int q = 0; q < static_cast<int>(rule.size()); q++) {
	r_elem = eval_quadrature<g, test, trial, geometry_dim, spatial_dim, Q, derivatives_type, lambda>(e, q, u_elem, r_elem, r, derivatives_ptr, J, X, num_elements, qf);
      }

      // once we've finished the element integration loop, write our element residuals
      // out to memory, to be later assembled into global residuals by mfem
      detail::Add(r, r_elem, e);
    } // e branch

  }

  template <Geometry g, typename test, typename trial, int geometry_dim, int spatial_dim, int Q, typename derivatives_type, typename lambda>
  __global__ void eval_cuda_quadrature(const mfem::DeviceTensor<2, const double> u, mfem::DeviceTensor<2, double> r, derivatives_type * derivatives_ptr, const mfem::DeviceTensor<4, const double> J, const mfem::DeviceTensor<3, const double> X, int num_elements, lambda qf) {

    using test_element               = finite_element<g, test>;
    using trial_element              = finite_element<g, trial>;
    using element_residual_type      = typename trial_element::residual_type;
    static constexpr auto rule       = GaussQuadratureRule<g, Q>();

    int qe = blockIdx.x * blockDim.x + threadIdx.x;
    // warps won't fetch that many elements ... not great.. but not horrible
    int e = qe / rule.size();
    int q = qe % rule.size();
    // for each element in the domain
    if (qe < num_elements * rule.size()) {
      // get the DOF values for this particular element
      auto u_elem = detail::Load<trial_element>(u, e);

      // this is where we will accumulate the element residual tensor
      element_residual_type r_elem{};

      // for each quadrature point in the element
      r_elem = eval_quadrature<g, test, trial, geometry_dim, spatial_dim, Q, derivatives_type, lambda>(e, q, u_elem, r_elem, r, derivatives_ptr, J, X, num_elements, qf);


      // once we've finished the element integration loop, write our element residuals
      // out to memory, to be later assembled into global residuals by mfem
      detail::Add(r, r_elem, e);
    } // qe branch

  }


template <Geometry g, typename test, typename trial, int geometry_dim, int spatial_dim, int Q,
typename derivatives_type, typename lambda>
void evaluation_kernel_cuda(const mfem::Vector& U, mfem::Vector& R, derivatives_type* derivatives_ptr,
const mfem::Vector& J_, const mfem::Vector& X_, int num_elements, lambda qf)
{

  using test_element               = finite_element<g, test>;
  using trial_element              = finite_element<g, trial>;
  using element_residual_type      = typename trial_element::residual_type;
  static constexpr int  test_ndof  = test_element::ndof;
  static constexpr int  trial_ndof = trial_element::ndof;
  static constexpr auto rule       = GaussQuadratureRule<g, Q>();

  // Use the device (GPU)
  X_.UseDevice(true);
  J_.UseDevice(true);
  U.UseDevice(true);
  R.UseDevice(true);

  // Note: Since we cannot call Reshape (__host__) within a kernel we pass in the resulting mfem::DeviceTensors which should be pointing to Device pointers via .Read() and .ReadWrite()

  // mfem provides this information in 1D arrays, so we reshape it
  // into strided multidimensional arrays before using
  auto X = detail::Reshape(X_.Read(), rule.size(), spatial_dim, num_elements);
  auto J = detail::Reshape(J_.Read(), rule.size(), spatial_dim, geometry_dim, num_elements);
  auto u = detail::Reshape<trial>(U.Read(), trial_ndof, num_elements);
  auto r = detail::Reshape<test>(R.ReadWrite(), test_ndof, num_elements);

  cudaDeviceSynchronize();
  serac::detail::displayLastCUDAErrorMessage(std::cout, "integral_cuda.cuh before eval_cuda is fine");

  const int blocksize = 128;
  [[maybe_unused]] int blocks_element = (num_elements +blocksize - 1)/blocksize;
  // eval_cuda_element<g, test, trial, geometry_dim, spatial_dim, Q ><<<blocks_element,blocksize>>>(u, r, derivatives_ptr, J, X, num_elements, qf);
  int blocks_quadrature_element = (num_elements * rule.size() + blocksize - 1)/blocksize;
  eval_cuda_quadrature<g, test, trial, geometry_dim, spatial_dim, Q ><<<blocks_quadrature_element,blocksize>>>(u, r, derivatives_ptr, J, X, num_elements, qf);

  cudaDeviceSynchronize();
  serac::detail::displayLastCUDAErrorMessage(std::cout, "integral_cuda.cuh after eval_cuda is fine");

  // copy back to host?
  R.HostRead();

  std::cout << "Host Read:" << std::endl;

  X_.UseDevice(false);
  J_.UseDevice(false);
  U.UseDevice(false);
  R.UseDevice(false);

}

}
