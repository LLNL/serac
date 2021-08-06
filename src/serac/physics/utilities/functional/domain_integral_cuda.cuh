#pragma once

#include "mfem.hpp"
#include "mfem/linalg/dtensor.hpp"

#include "serac/physics/utilities/functional/integral_utilities.hpp"
#include "serac/physics/utilities/functional/domain_integral_shared.hpp"
#include <cstring>

namespace serac {

namespace detail {

/**
 * @brief utility method to display last cuda error message
 *
 * @param[in] o The output stream to post success or CUDA error messages
 * @param[in] success_string A string to print if there are no CUDA error messages
 */
inline void displayLastCUDAErrorMessage(std::ostream& o, const char* success_string = "")
{
  auto error = cudaGetLastError();
  if (error != cudaError::cudaSuccess) {
    o << "Last CUDA Error Message :" << cudaGetErrorString(error) << std::endl;
  } else if (strlen(success_string) > 0) {
    o << success_string << std::endl;
  }
}

/**
 * @brief Defines whether to loop by elements or quadrature points
 */
enum ThreadExecutionPolicy
{
  THREAD_PER_ELEMENT_QUADRATURE_POINT,
  THREAD_PER_ELEMENT
};

/**
 * @brief Contains the GPU launch configuration
 */
struct ThreadExecutionConfiguration {
  int blocksize;
};

}  // namespace detail

namespace domain_integral {

template <Geometry g, typename test, typename trial, int Q, typename derivatives_type, typename lambda, typename u_type,
          typename r_type, typename J_type, typename X_type>
__global__ void eval_cuda_element(const u_type u, r_type r, derivatives_type* derivatives_ptr, J_type J, X_type X,
                                  int num_elements, lambda qf)
{
  using test_element          = finite_element<g, test>;
  using trial_element         = finite_element<g, trial>;
  using element_residual_type = typename trial_element::residual_type;
  static constexpr auto rule  = GaussQuadratureRule<g, Q>();

  // for each element in the domain
  const int grid_stride = blockDim.x * gridDim.x;
#pragma unroll
  for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < num_elements; e += grid_stride) {
    // get the DOF values for this particular element
    auto u_elem = detail::Load<trial_element>(u, e);

    // this is where we will accumulate the element residual tensor
    element_residual_type r_elem{};

    // for each quadrature point in the element
    for (int q = 0; q < static_cast<int>(rule.size()); q++) {
      eval_quadrature<g, test, trial, Q, derivatives_type, lambda>(e, q, u_elem, r_elem, derivatives_ptr, J, X,
                                                                   num_elements, qf);
    }

    // once we've finished the element integration loop, write our element residuals
    // out to memory, to be later assembled into global residuals by mfem
    detail::Add(r, r_elem, e);
  }  // e loop
}

template <Geometry g, typename test, typename trial, int Q, typename derivatives_type, typename lambda, typename u_type,
          typename r_type, typename J_type, typename X_type>
__global__ void eval_cuda_quadrature(const u_type u, r_type r, derivatives_type* derivatives_ptr, J_type & J, X_type & X, int num_elements, lambda qf)
{
  using test_element          = finite_element<g, test>;
  using trial_element         = finite_element<g, trial>;
  using element_residual_type = typename trial_element::residual_type;
  static constexpr auto rule  = GaussQuadratureRule<g, Q>();

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
    eval_quadrature<g, test, trial, Q, derivatives_type, lambda>(e, q, u_elem, r_elem, derivatives_ptr, J, X,
                                                                 num_elements, qf);

    // once we've finished the element integration loop, write our element residuals
    // out to memory, to be later assembled into global residuals by mfem
    detail::Add(r, r_elem, e);
  }  // quadrature x element loop
}

template <Geometry g, typename test, typename trial, int Q, serac::detail::ThreadExecutionPolicy policy,
          typename derivatives_type, typename lambda>
void evaluation_kernel_cuda(serac::detail::ThreadExecutionConfiguration config, const mfem::Vector U, mfem::Vector R,
                            derivatives_type* derivatives_ptr, const mfem::Vector& J_, const mfem::Vector& X_,
                            int num_elements, lambda qf)
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

  // Note: Since we cannot call Reshape (__host__) within a kernel we pass in the resulting mfem::DeviceTensors which
  // should be pointing to Device pointers via .Read() and .ReadWrite()

  // mfem provides this information in 1D arrays, so we reshape it
  // into strided multidimensional arrays before using
  auto X = mfem::Reshape(X_.Read(), rule.size(), dim, num_elements);
  auto J = mfem::Reshape(J_.Read(), rule.size(), dim, dim, num_elements);
  auto u = detail::Reshape<trial>(U.Read(), trial_ndof, num_elements);
  auto r = detail::Reshape<test>(R.ReadWrite(), test_ndof, num_elements);

  std::cout << "X:" << &X << " " << X_.Read() << std::endl;
  std::cout << "J:" << &J << " " << J_.Read() << std::endl;
  std::cout << "u:" << &u << " " << U.Read() << std::endl;
  std::cout << "r:" << &r << " " << R.ReadWrite() << std::endl;
  std::cout << "qf:" << &qf << " " << std::endl;
  
  cudaDeviceSynchronize();
  serac::detail::displayLastCUDAErrorMessage(std::cout);

  if constexpr (policy == serac::detail::ThreadExecutionPolicy::THREAD_PER_ELEMENT_QUADRATURE_POINT) {
    int blocks_quadrature_element = (num_elements * rule.size() + config.blocksize - 1) / config.blocksize;
    eval_cuda_quadrature<g, test, trial, Q>
        <<<blocks_quadrature_element, config.blocksize>>>(u, r, derivatives_ptr, J, X, num_elements, qf);

  } else if constexpr (policy == serac::detail::ThreadExecutionPolicy::THREAD_PER_ELEMENT) {
    int blocks_element = (num_elements + config.blocksize - 1) / config.blocksize;
    eval_cuda_element<g, test, trial, Q>
        <<<blocks_element, config.blocksize>>>(u, r, derivatives_ptr, J, X, num_elements, qf);
  }

  cudaDeviceSynchronize();
  serac::detail::displayLastCUDAErrorMessage(std::cout);

  std::vector<double> debug_cuda(test_ndof, num_elements);
  cudaMemcpy(debug_cuda.data(), R.ReadWrite(), debug_cuda.size() * sizeof(double), cudaMemcpyDeviceToHost);
  
  // copy back to host
  R.HostRead();

  // X_.UseDevice(false);
  // J_.UseDevice(false);
  // U.UseDevice(false);
  // R.UseDevice(false);
}

template <Geometry g, typename test, typename trial, int Q, typename derivatives_type, typename du_type,
          typename dr_type>
__global__ void gradient_cuda_element(const du_type du, dr_type dr, derivatives_type* derivatives_ptr,
                                      const mfem::DeviceTensor<4, const double> J, int num_elements)
{
  using test_element          = finite_element<g, test>;
  using trial_element         = finite_element<g, trial>;
  using element_residual_type = typename trial_element::residual_type;
  static constexpr auto rule  = GaussQuadratureRule<g, Q>();

  const int grid_stride = blockDim.x * gridDim.x;
#pragma unroll
  for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < num_elements; e += grid_stride) {
    // get the (change in) values for this particular element
    tensor du_elem = detail::Load<trial_element>(du, e);

    // this is where we will accumulate the (change in) element residual tensor
    element_residual_type dr_elem{};

    // for each quadrature point in the element
    for (int q = 0; q < static_cast<int>(rule.size()); q++) {
      gradient_quadrature<g, test, trial, Q, derivatives_type>(e, q, du_elem, dr_elem, derivatives_ptr, J,
                                                               num_elements);
    }

    // once we've finished the element integration loop, write our element residuals
    // out to memory, to be later assembled into global residuals by mfem
    detail::Add(dr, dr_elem, e);
  }
}

template <Geometry g, typename test, typename trial, int Q, typename derivatives_type, typename du_type,
          typename dr_type>
__global__ void gradient_cuda_quadrature(const du_type du, dr_type dr, derivatives_type* derivatives_ptr,
                                         const mfem::DeviceTensor<4, const double> J, int num_elements)
{
  using test_element          = finite_element<g, test>;
  using trial_element         = finite_element<g, trial>;
  using element_residual_type = typename trial_element::residual_type;
  static constexpr auto rule  = GaussQuadratureRule<g, Q>();

  const int grid_stride           = blockDim.x * gridDim.x;
  auto      thread_id             = blockIdx.x * blockDim.x + threadIdx.x;
  auto      num_quadrature_points = num_elements * rule.size();
#pragma unroll
  for (int qe = thread_id; qe < num_quadrature_points; qe += grid_stride) {
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

template <Geometry g, typename test, typename trial, int Q, serac::detail::ThreadExecutionPolicy policy,
          typename derivatives_type>
void gradient_kernel_cuda(serac::detail::ThreadExecutionConfiguration config, const mfem::Vector& dU, mfem::Vector& dR,
                          derivatives_type* derivatives_ptr, const mfem::Vector& J_, int num_elements)
{
  using test_element               = finite_element<g, test>;
  using trial_element              = finite_element<g, trial>;
  using element_residual_type      = typename trial_element::residual_type;
  static constexpr int  test_ndof  = test_element::ndof;
  static constexpr int  trial_ndof = trial_element::ndof;
  static constexpr auto rule       = GaussQuadratureRule<g, Q>();
  static constexpr int  dim        = dimension_of(g);

  // Use the device (GPU)
  // J_.UseDevice(true);
  // dU.UseDevice(true);
  // dR.UseDevice(true);

  // mfem provides this information in 1D arrays, so we reshape it
  // into strided multidimensional arrays before using
  auto J  = mfem::Reshape(J_.Read(), rule.size(), dim, dim, num_elements);
  auto du = detail::Reshape<trial>(dU.Read(), trial_ndof, num_elements);
  auto dr = detail::Reshape<test>(dR.ReadWrite(), test_ndof, num_elements);

  cudaDeviceSynchronize();
  serac::detail::displayLastCUDAErrorMessage(std::cout);

  // call gradient_cuda
  if constexpr (policy == serac::detail::ThreadExecutionPolicy::THREAD_PER_ELEMENT_QUADRATURE_POINT) {
    int blocks_quadrature_element = (num_elements * rule.size() + config.blocksize - 1) / config.blocksize;
    gradient_cuda_quadrature<g, test, trial, Q, derivatives_type>
        <<<blocks_quadrature_element, config.blocksize>>>(du, dr, derivatives_ptr, J, num_elements);

  } else if constexpr (policy == serac::detail::ThreadExecutionPolicy::THREAD_PER_ELEMENT) {
    int blocks_element = (num_elements + config.blocksize - 1) / config.blocksize;
    gradient_cuda_element<g, test, trial, Q, derivatives_type>
        <<<blocks_element, config.blocksize>>>(du, dr, derivatives_ptr, J, num_elements);
  }

  cudaDeviceSynchronize();
  serac::detail::displayLastCUDAErrorMessage(std::cout);
  dR.HostRead();

  // J_.UseDevice(false);
  // dU.UseDevice(false);
  // dR.UseDevice(false);
}

}  // namespace domain_integral

}  // namespace serac
