#pragma once

#include "serac/physics/utilities/functional/integral.hpp"

#if defined(__CUDACC__)
#include "serac/physics/utilities/functional/integral_cuda.cuh"
#endif

namespace serac {

/**
 * @brief Describes a single integral term in a weak forumulation of a partial differential equation
 * @tparam spaces A @p std::function -like set of template parameters that describe the test and trial
 * function spaces, i.e., @p test(trial)
 */
template <typename spaces>
class Integral {
public:
  using test_space  = test_space_t<spaces>;   ///< the test function space
  using trial_space = trial_space_t<spaces>;  ///< the trial function space

  /**
   * @brief Constructs an @p Integral from a user-provided quadrature function
   * @tparam geometry_dim The dimension of the element (2 for quad, 3 for hex, etc)
   * @tparam spatial_dim The full dimension of the mesh
   * @param[in] num_elements The number of elements in the mesh
   * @param[in] J The Jacobians of the element transformations at all quadrature points
   * @param[in] X The actual (not reference) coordinates of all quadrature points
   * @see mfem::GeometricFactors
   * @param[in] qf The user-provided quadrature function
   * @note The @p Dimension parameters are used to assist in the deduction of the @a geometry_dim
   * and @a spatial_dim template parameters
   */
  template <int geometry_dim, int spatial_dim, typename lambda_type>
  Integral(int num_elements, const mfem::Vector& J, const mfem::Vector& X, Dimension<geometry_dim>,
           Dimension<spatial_dim>, lambda_type&& qf, bool use_cuda = false)
      : J_(J), X_(X)
  {
    constexpr auto geometry                      = supported_geometries[geometry_dim];
    constexpr auto Q                             = std::max(test_space::order, trial_space::order) + 1;
    constexpr auto quadrature_points_per_element = (spatial_dim == 2) ? Q * Q : Q * Q * Q;

    uint32_t num_quadrature_points = quadrature_points_per_element * uint32_t(num_elements);

    // these lines of code figure out the argument types that will be passed
    // into the quadrature function in the finite element kernel.
    //
    // we use them to observe the output type and allocate memory to store
    // the derivative information at each quadrature point
    using x_t             = tensor<double, spatial_dim>;
    using u_du_t          = typename detail::lambda_argument<trial_space, geometry_dim, spatial_dim>::type;
    using derivative_type = decltype(get_gradient(qf(x_t{}, make_dual(u_du_t{}))));

    // the derivative_type data is stored in a shared_ptr here, because it can't be a
    // member variable on the Integral class template (since it depends on the lambda function,
    // which isn't known until the time of construction).
    //
    // This shared_ptr should have a comparable lifetime to the Integral instance itself, since
    // the reference count will increase when it is captured by the lambda functions below, and
    // the reference count will go back to zero after those std::functions are deconstructed in
    // Integral::~Integral()
    //
    // derivatives are stored as a 2D array, such that quadrature point q of element e is accessed by
    // qf_derivatives[e * quadrature_points_per_element + q]
    std::shared_ptr<derivative_type[]> qf_derivatives(new derivative_type[num_quadrature_points]);

    // this is where we actually specialize the finite element kernel templates with
    // our specific requirements (element type, test/trial spaces, quadrature rule, q-function, etc).
    //
    // std::function's type erasure lets us wrap those specific details inside a function with known signature
    //
    // note: the qf_derivatives_ptr is copied by value to each lambda function below,
    //       to allow the evaluation kernel to pass derivative values to the gradient kernel
    evaluation_ = [=](const mfem::Vector& U, mfem::Vector& R) {
      if (use_cuda) {
#if defined(__CUDACC__)
	evaluation_kernel_cuda<geometry, test_space, trial_space, geometry_dim, spatial_dim, Q><<<1,1>>>(U, R, qf_derivatives.get(), J_,
                                                                                         X_, num_elements, qf);
      #endif
	} else {
      evaluation_kernel<geometry, test_space, trial_space, geometry_dim, spatial_dim, Q>(U, R, qf_derivatives.get(), J_,
                                                                                         X_, num_elements, qf);
      }
    };

    gradient_ = [=](const mfem::Vector& dU, mfem::Vector& dR) {
      gradient_kernel<geometry, test_space, trial_space, geometry_dim, spatial_dim, Q>(dU, dR, qf_derivatives.get(), J_,
                                                                                       num_elements);
    };
  }

  /**
   * @brief Applies the integral, i.e., @a output_E = evaluate( @a input_E )
   * @param[in] input_E The input to the evaluation; per-element DOF values
   * @param[out] output_E The output of the evalution; per-element DOF residuals
   * @see evaluation_kernel
   */
  void Mult(const mfem::Vector& input_E, mfem::Vector& output_E) const { evaluation_(input_E, output_E); }

  /**
   * @brief Applies the integral, i.e., @a output_E = gradient( @a input_E )
   * @param[in] input_E The input to the evaluation; per-element DOF values
   * @param[out] output_E The output of the evalution; per-element DOF residuals
   * @see gradient_kernel
   */
  void GradientMult(const mfem::Vector& input_E, mfem::Vector& output_E) const { gradient_(input_E, output_E); }

private:
  /**
   * @brief Jacobians of the element transformations at all quadrature points
   */
  const mfem::Vector J_;
  /**
   * @brief Mapped (physical) coordinates of all quadrature points
   */
  const mfem::Vector X_;

  /**
   * @brief Type-erased handle to evaluation kernel
   * @see evaluation_kernel
   */
  std::function<void(const mfem::Vector&, mfem::Vector&)> evaluation_;
  /**
   * @brief Type-erased handle to gradient kernel
   * @see gradient_kernel
   */
  std::function<void(const mfem::Vector&, mfem::Vector&)> gradient_;
};

}
