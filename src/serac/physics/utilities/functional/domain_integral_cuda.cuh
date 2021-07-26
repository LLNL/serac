#pragma once

#include "mfem.hpp"
#include "mfem/linalg/dtensor.hpp"

#include "serac/physics/utilities/functional/integral_utilities.hpp"
#include "serac/physics/utilities/functional/domain_integral_shared.hpp"
#include <cstring>

namespace serac {

namespace detail {
template <typename T, typename... Dims>
__host__ inline mfem::DeviceTensor<sizeof...(Dims),T> Reshape(T *ptr, Dims... dims)
   {
      return mfem::DeviceTensor<sizeof...(Dims),T>(ptr, dims...);
   }


  /**
   * @brief utility method to display last cuda error message
   *
   * @param[in] o The output stream to post success or CUDA error messages
   * @param[in] success_string A string to print if there are no CUDA error messages
   */
  inline void displayLastCUDAErrorMessage(std::ostream & o, const char * success_string = "") {
      auto error = cudaGetLastError();
      if(error != cudaError::cudaSuccess) {
	o << "Last CUDA Error Message :" << cudaGetErrorString(error) << std::endl;
      } else if (strlen(success_string) > 0) {
	o << success_string << std::endl;
      }
    }

} // namespace detail

  namespace domain_integral {  

  template <Geometry g, typename test, typename trial, int Q, typename derivatives_type, typename lambda, typename u_type, typename r_type, typename J_type, typename X_type>
  __global__ void eval_cuda_element(const u_type u, r_type r, derivatives_type * derivatives_ptr, J_type J, X_type X, int num_elements, lambda qf) {

    using test_element               = finite_element<g, test>;
    using trial_element              = finite_element<g, trial>;
    using element_residual_type      = typename trial_element::residual_type;
    static constexpr auto rule       = GaussQuadratureRule<g, Q>();

    // for each element in the domain
    const int grid_stride = blockDim.x * gridDim.x;
#pragma unroll
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < num_elements ; e += grid_stride) {
      // get the DOF values for this particular element
      auto u_elem = detail::Load<trial_element>(u, e);

      // this is where we will accumulate the element residual tensor
      element_residual_type r_elem{};

      // for each quadrature point in the element
      for (int q = 0; q < static_cast<int>(rule.size()); q++) {
	eval_quadrature<g, test, trial, Q, derivatives_type, lambda>(e, q, u_elem, r_elem, derivatives_ptr, J, X, num_elements, qf);
      }

      // once we've finished the element integration loop, write our element residuals
      // out to memory, to be later assembled into global residuals by mfem
      detail::Add(r, r_elem, e);
    } // e loop

  }

  template <Geometry g, typename test, typename trial, int Q, typename derivatives_type, typename lambda, typename u_type, typename r_type, typename J_type, typename X_type>
  __global__ void eval_cuda_quadrature(const u_type u, r_type r, derivatives_type * derivatives_ptr, J_type J, X_type X, int num_elements, lambda qf) {

    using test_element               = finite_element<g, test>;
    using trial_element              = finite_element<g, trial>;
    using element_residual_type      = typename trial_element::residual_type;
    static constexpr auto rule       = GaussQuadratureRule<g, Q>();

    const int grid_stride = blockDim.x * gridDim.x;
    // launch a thread for each quadrature x element point
    for (int qe = blockIdx.x * blockDim.x + threadIdx.x; qe < num_elements * rule.size(); qe += grid_stride) {
    // warps won't fetch that many elements ... not great.. but not horrible
    int e = qe / rule.size();
    int q = qe % rule.size();
      // get the DOF values for this particular element
      auto u_elem = detail::Load<trial_element>(u, e);

      // this is where we will accumulate the element residual tensor
      element_residual_type r_elem{};

      // for each quadrature point in the element
      eval_quadrature<g, test, trial, Q, derivatives_type, lambda>(e, q, u_elem, r_elem, derivatives_ptr, J, X, num_elements, qf);

      // once we've finished the element integration loop, write our element residuals
      // out to memory, to be later assembled into global residuals by mfem
      detail::Add(r, r_elem, e);
    } // quadrature x element loop

  }


    template <Geometry g, typename test, typename trial, int Q, int blocksize,
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
  static constexpr int  dim        = dimension_of(g);
  
  // Use the device (GPU)
  X_.UseDevice(true);
  J_.UseDevice(true);
  U.UseDevice(true);
  R = 0.;
  R.UseDevice(true);

  // Note: Since we cannot call Reshape (__host__) within a kernel we pass in the resulting mfem::DeviceTensors which should be pointing to Device pointers via .Read() and .ReadWrite()

  // mfem provides this information in 1D arrays, so we reshape it
  // into strided multidimensional arrays before using
  auto X = detail::Reshape(X_.Read(), rule.size(), dim, num_elements);
  auto J = detail::Reshape(J_.Read(), rule.size(), dim, dim, num_elements);
  auto u = detail::Reshape<trial>(U.Read(), trial_ndof, num_elements);
  auto r = detail::Reshape<test>(R.ReadWrite(), test_ndof, num_elements);

  cudaDeviceSynchronize();
  serac::detail::displayLastCUDAErrorMessage(std::cout);

  [[maybe_unused]] int blocks_element = (num_elements + blocksize - 1)/blocksize;
  //  eval_cuda_element<g, test, trial, Q ><<<blocks_element,blocksize>>>(u, r, derivatives_ptr, J, X, num_elements, qf);
  int blocks_quadrature_element = (num_elements * rule.size() + blocksize - 1)/blocksize;
  eval_cuda_quadrature<g, test, trial, Q ><<<blocks_quadrature_element,blocksize>>>(u, r, derivatives_ptr, J, X, num_elements, qf);

  cudaDeviceSynchronize();
  serac::detail::displayLastCUDAErrorMessage(std::cout);

  // copy back to host
  R.HostRead();

  X_.UseDevice(false);
  J_.UseDevice(false);
  U.UseDevice(false);
  R.UseDevice(false);

}


template <Geometry g, typename test, typename trial, int Q,
          typename derivatives_type, typename du_elem_type, typename dr_elem_type>
SERAC_HOST_DEVICE void gradient_quadrature(int e, int q, du_elem_type & du_elem, dr_elem_type & dr_elem, derivatives_type* derivatives_ptr, const mfem::DeviceTensor<4, const double> J, int num_elements)
{

  using test_element               = finite_element<g, test>;
  using trial_element              = finite_element<g, trial>;
  static constexpr auto rule       = GaussQuadratureRule<g, Q>();
  static constexpr int  dim        = dimension_of(g);
  
  // get the position of this quadrature point in the parent and physical space,
  // and calculate the measure of that point in physical space.
  auto   xi  = rule.points[q];
  auto   dxi = rule.weights[q];
  auto   J_q = make_tensor<dim, dim>([&](int i, int j) { return J(q, i, j, e); });
  double dx  = det(J_q) * dxi;

  // evaluate the (change in) value/derivatives at this quadrature point
  auto darg = Preprocess<trial_element>(du_elem, xi, J_q);

  // recall the derivative of the q-function w.r.t. its arguments at this quadrature point

  auto dq_darg = detail::AccessDerivatives(derivatives_ptr, e, q, rule, num_elements);

  // use the chain rule to compute the first-order change in the q-function output
  auto dq = chain_rule(dq_darg, darg);

  // integrate dq against test space shape functions / gradients
  // to get the (change in) element residual contributions
  dr_elem += Postprocess<test_element>(dq, xi, J_q) * dx;

}
  
  
template <Geometry g, typename test, typename trial, int Q,
          typename derivatives_type, typename du_type, typename dr_type>
__global__ void gradient_cuda_element(const du_type du, dr_type dr, derivatives_type* derivatives_ptr,
                     const mfem::DeviceTensor<4, const double> J, int num_elements)
{
  using test_element               = finite_element<g, test>;
  using trial_element              = finite_element<g, trial>;
  using element_residual_type      = typename trial_element::residual_type;
  static constexpr auto rule       = GaussQuadratureRule<g, Q>();


  // for each element in the domain
  //  for (int e = 0; e < num_elements; e++) {
  //int e = blockIdx.x * blockDim.x + threadIdx.x;
  //if (e < num_elements) {
  const int grid_stride = blockDim.x * gridDim.x;
#pragma unroll
  for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < num_elements; e += grid_stride) {
    // get the (change in) values for this particular element
    tensor du_elem = detail::Load<trial_element>(du, e);

    // this is where we will accumulate the (change in) element residual tensor
    element_residual_type dr_elem{};

    // for each quadrature point in the element
    for (int q = 0; q < static_cast<int>(rule.size()); q++) {

      gradient_quadrature<g, test, trial, Q, derivatives_type>(e, q, du_elem, dr_elem, derivatives_ptr, J, num_elements);
    }

    // once we've finished the element integration loop, write our element residuals
    // out to memory, to be later assembled into global residuals by mfem
    detail::Add(dr, dr_elem, e);
  }
}

template <Geometry g, typename test, typename trial, int Q,
          typename derivatives_type, typename du_type, typename dr_type>
__global__ void gradient_cuda_quadrature(const du_type du, dr_type dr, derivatives_type* derivatives_ptr,
                     const mfem::DeviceTensor<4, const double> J, int num_elements)
{
  using test_element               = finite_element<g, test>;
  using trial_element              = finite_element<g, trial>;
  using element_residual_type      = typename trial_element::residual_type;
  static constexpr auto rule       = GaussQuadratureRule<g, Q>();


  // for each element in the domain
  //  for (int e = 0; e < num_elements; e++) {
  // int qe = blockIdx.x * blockDim.x + threadIdx.x;
  // if (qe < num_elements * rule.size()) {

  const int grid_stride = blockDim.x * gridDim.x;
#pragma unroll
  for (int qe = blockIdx.x * blockDim.x + threadIdx.x; qe < num_elements * rule.size(); qe += grid_stride) {

    int e = qe / rule.size();
    int q = qe % rule.size();
    // get the (change in) values for this particular element
    tensor du_elem = detail::Load<trial_element>(du, e);

    // this is where we will accumulate the (change in) element residual tensor
    element_residual_type dr_elem{};

    gradient_quadrature<g, test, trial, Q, derivatives_type>(e, q, du_elem, dr_elem, derivatives_ptr, J, num_elements);

    // once we've finished the element integration loop, write our element residuals
    // out to memory, to be later assembled into global residuals by mfem
    detail::Add(dr, dr_elem, e);
  }
}

  

template <Geometry g, typename test, typename trial, int Q,
          typename derivatives_type>
void gradient_kernel_cuda(const mfem::Vector& dU, mfem::Vector& dR, derivatives_type* derivatives_ptr,
                     const mfem::Vector& J_, int num_elements)
{
  using test_element               = finite_element<g, test>;
  using trial_element              = finite_element<g, trial>;
  using element_residual_type      = typename trial_element::residual_type;
  static constexpr int  test_ndof  = test_element::ndof;
  static constexpr int  trial_ndof = trial_element::ndof;
  static constexpr auto rule       = GaussQuadratureRule<g, Q>();
  static constexpr int  dim        = dimension_of(g);
  
  // Use the device (GPU)
  J_.UseDevice(true);
  dU.UseDevice(true);
  dR.UseDevice(true);

  // mfem provides this information in 1D arrays, so we reshape it
  // into strided multidimensional arrays before using
  auto J  = mfem::Reshape(J_.Read(), rule.size(), dim, dim, num_elements);
  auto du = detail::Reshape<trial>(dU.Read(), trial_ndof, num_elements);
  auto dr = detail::Reshape<test>(dR.ReadWrite(), test_ndof, num_elements);

  cudaDeviceSynchronize();
  serac::detail::displayLastCUDAErrorMessage(std::cout);

  // call gradient_cuda
  const int blocksize = 128;
  [[maybe_unused]] int blocks_element = (num_elements + blocksize - 1)/blocksize;
  // gradient_cuda_element<g, test, trial, Q, derivatives_type> <<< blocks_element, blocksize>>>(du, dr, derivatives_ptr, J, num_elements);

  [[maybe_unused]] int blocks_quadrature_element = (num_elements * rule.size() + blocksize - 1)/blocksize;
  gradient_cuda_quadrature<g, test, trial, Q, derivatives_type> <<< blocks_quadrature_element, blocksize>>>(du, dr, derivatives_ptr, J, num_elements);

  
  cudaDeviceSynchronize();
  serac::detail::displayLastCUDAErrorMessage(std::cout);
  dR.HostRead();

  J_.UseDevice(false);
  dU.UseDevice(false);
  dR.UseDevice(false);


}

} // namespace domain_integral

} // namespace serac
