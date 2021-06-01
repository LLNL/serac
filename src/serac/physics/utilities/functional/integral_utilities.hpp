#pragma once

#include "mfem.hpp"
#include "mfem/linalg/dtensor.hpp"

#include "serac/physics/utilities/functional/tensor.hpp"
#include "serac/physics/utilities/functional/finite_element.hpp"

namespace serac {

namespace detail {

/**
 * @brief Wrapper for mfem::Reshape compatible with Serac's finite element space types
 * @tparam space A test or trial space
 * @param[in] u The raw data to reshape into an mfem::DeviceTensor
 * @param[in] n1 The first dimension to reshape into
 * @param[in] n2 The second dimension to reshape into
 * @see mfem::Reshape
 */
template <typename space>
auto Reshape(double* u, int n1, int n2)
{
  if constexpr (space::components == 1) {
    return mfem::Reshape(u, n1, n2);
  }
  if constexpr (space::components > 1) {
    return mfem::Reshape(u, n1, space::components, n2);
  }
};

/// @overload
template <typename space>
auto Reshape(const double* u, int n1, int n2)
{
  if constexpr (space::components == 1) {
    return mfem::Reshape(u, n1, n2);
  }
  if constexpr (space::components > 1) {
    return mfem::Reshape(u, n1, space::components, n2);
  }
};

/**
 * @brief Extracts the dof values for a particular element
 * @param[in] u The decomposed per-element DOFs, libCEED's E-vector
 * @param[in] e The index of the element to retrieve DOFs for
 * @note For the case of only 1 dof per node, detail::Load returns a tensor<double, ndof>
 */
template <int ndof>
inline auto Load(const mfem::DeviceTensor<2, const double>& u, int e)
{
  return make_tensor<ndof>([&u, e](int i) { return u(i, e); });
}

/**
 * @overload
 * @note For the case of multiple dofs per node, detail::Load returns a tensor<double, components, ndof>
 */
template <int ndof, int components>
inline auto Load(const mfem::DeviceTensor<3, const double>& u, int e)
{
  return make_tensor<components, ndof>([&u, e](int j, int i) { return u(i, j, e); });
}

/**
 * @overload
 * @note Intended to be used with Serac's finite element space types
 */
template <typename space, typename T>
auto Load(const T& u, int e)
{
  if constexpr (space::components == 1) {
    return detail::Load<space::ndof>(u, e);
  }
  if constexpr (space::components > 1) {
    return detail::Load<space::ndof, space::components>(u, e);
  }
};

/**
 * @brief Adds the contributions of the local residual to the global residual
 * @param[inout] r_global The full element-decomposed residual
 * @param[in] r_local The contributions to a residual from a single element
 * @param[in] e The index of the element whose residual is @a r_local
 */
template <int ndof>
void Add(const mfem::DeviceTensor<2, double>& r_global, tensor<double, ndof> r_local, int e)
{
  for (int i = 0; i < ndof; i++) {
    r_global(i, e) += r_local[i];
  }
}

/**
 * @overload
 * @note Used when each node has multiple DOFs
 */
template <int ndof, int components>
void Add(const mfem::DeviceTensor<3, double>& r_global, tensor<double, ndof, components> r_local, int e)
{
  for (int i = 0; i < ndof; i++) {
    for (int j = 0; j < components; j++) {
      r_global(i, j, e) += r_local[i][j];
    }
  }
}


/**
 * @brief a class that helps to extract the test space from a function signature template parameter
 * @tparam space The function signature itself
 */
template <typename spaces>
struct get_test_space;  // undefined

/**
 * @brief a class that helps to extract the test space from a function signature template parameter
 * @tparam space The function signature itself
 */
template <typename test_space, typename trial_space>
struct get_test_space<test_space(trial_space)> {
  using type = test_space;  ///< the test space
};

/**
 * @brief a class that helps to extract the trial space from a function signature template parameter
 * @tparam space The function signature itself
 */
template <typename spaces>
struct get_trial_space;  // undefined

/**
 * @brief a class that helps to extract the trial space from a function signature template parameter
 * @tparam space The function signature itself
 */
template <typename test_space, typename trial_space>
struct get_trial_space<test_space(trial_space)> {
  using type = trial_space;  ///< the trial space
};

/**
 * @brief a class that provides the lambda argument types for a given integral
 * @tparam trial_space the trial space associated with the integral
 * @tparam geometry_dim the dimensionality of the element type
 * @tparam spatial_dim the dimensionality of the space the mesh lives in
 */
template <typename space, int geometry_dim, int spatial_dim>
struct lambda_argument;

// specialization for an H1 space with polynomial order p, and c components
template <int p, int c, int dim>
struct lambda_argument<H1<p, c>, dim, dim> {
  using type = std::tuple<reduced_tensor<double, c>, reduced_tensor<double, c, dim> >;
};

// specialization for an L2 space with polynomial order p, and c components
template <int p, int c, int dim>
struct lambda_argument<L2<p, c>, dim, dim> {
  using type = std::tuple<reduced_tensor<double, c>, reduced_tensor<double, c, dim> >;
};

// specialization for an H1 space with polynomial order p, and c components
// evaluated in a line integral or surface integral. Note: only values are provided in this case
template <int p, int c, int geometry_dim, int spatial_dim>
struct lambda_argument<H1<p, c>, geometry_dim, spatial_dim> {
  using type = reduced_tensor<double, c>;
};

// specialization for an Hcurl space with polynomial order p in 2D
template <int p>
struct lambda_argument<Hcurl<p>, 2, 2> {
  using type = std::tuple<tensor<double, 2>, double>;
};

// specialization for an Hcurl space with polynomial order p in 3D
template <int p>
struct lambda_argument<Hcurl<p>, 3, 3> {
  using type = std::tuple<tensor<double, 3>, tensor<double, 3> >;
};

}

/**
 * @brief a type function that extracts the test space from a function signature template parameter
 * @tparam space The function signature itself
 */
template <typename spaces>
using test_space_t = typename detail::get_test_space<spaces>::type;

/**
 * @brief a type function that extracts the trial space from a function signature template parameter
 * @tparam space The function signature itself
 */
template <typename spaces>
using trial_space_t = typename detail::get_trial_space<spaces>::type;

static constexpr Geometry supported_geometries[] = {Geometry::Point, Geometry::Segment, Geometry::Quadrilateral,
                                                    Geometry::Hexahedron};

}
