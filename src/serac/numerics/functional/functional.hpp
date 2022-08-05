// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file functional.hpp
 *
 * @brief Implementation of the quadrature-function-based functional enabling rapid development of FEM formulations
 */

#pragma once

#include "mfem.hpp"

#include "serac/infrastructure/logger.hpp"
#include "serac/numerics/functional/tensor.hpp"
#include "serac/numerics/functional/quadrature.hpp"
#include "serac/numerics/functional/finite_element.hpp"
#include "serac/numerics/functional/tuple_arithmetic.hpp"
#include "serac/numerics/functional/domain_integral.hpp"
#include "serac/numerics/functional/boundary_integral.hpp"
#include "serac/numerics/functional/dof_numbering.hpp"

namespace serac {

template <int i>
struct DifferentiateWRT {
};

/**
 * @brief this type exists solely as a way to signal to `serac::Functional` that the function
 * serac::Functional::operator()` should differentiate w.r.t. a specific argument
 */
struct differentiate_wrt_this {
  const mfem::Vector& ref;  ///< the actual data wrapped by this type

  /// @brief implicitly convert back to `mfem::Vector` to extract the actual data
  operator const mfem::Vector &() const { return ref; }
};

/**
 * @brief this function is intended to only be used in combination with
 *   `serac::Functional::operator()`, as a way for the user to express that
 *   it should both evaluate and differentiate w.r.t. a specific argument (only 1 argument at a time)
 *
 * For example:
 * @code{.cpp}
 *     mfem::Vector arg0 = ...;
 *     mfem::Vector arg1 = ...;
 *     mfem::Vector just_the_value = my_functional(arg0, arg1);
 *     auto [value, gradient_wrt_arg1] = my_functional(arg0, differentiate_wrt(arg1));
 * @endcode
 */
auto differentiate_wrt(const mfem::Vector& v) { return differentiate_wrt_this{v}; }

/**
 * @tparam T a list of types, containing at most 1 `differentiate_wrt_this`
 *
 * @brief given a list of types, this function returns the index that corresponds to the type `dual_vector`.
 *
 * e.g.
 * @code{.cpp}
 * static_assert(index_of_dual_vector < foo, bar, differentiate_wrt_this, baz, qux >() == 2);
 * @endcode
 */
template <typename... T>
constexpr int index_of_differentiation()
{
  constexpr int n          = int(sizeof...(T));
  bool          matching[] = {std::is_same_v<T, differentiate_wrt_this>...};
  for (int i = 0; i < n; i++) {
    if (matching[i]) {
      return i;
    }
  }
  return -1;
}

/**
 * @brief Compile-time alias for index of differentiation
 */
template <int ind>
struct Index {
  /**
   * @brief Returns the index
   */
  constexpr operator int() { return ind; }
};

/// @cond
template <typename T, ExecutionSpace exec = serac::default_execution_space>
class Functional;
/// @endcond

/**
 * @brief Intended to be like @p std::function for finite element kernels
 *
 * That is: you tell it the inputs (trial spaces) for a kernel, and the outputs (test space) like @p std::function
 * For example, this code represents a function that takes an integer argument and returns a double:
 * @code{.cpp}
 * std::function< double(double, int) > my_func;
 * @endcode
 * And this represents a function that takes values from an Hcurl field and returns a
 * residual vector associated with an H1 field:
 * @code{.cpp}
 * Functional< H1(Hcurl) > my_residual;
 * @endcode
 *
 * @tparam test The space of test functions to use
 * @tparam trial The space of trial functions to use
 * @tparam exec whether to carry out calculations on CPU or GPU
 *
 * To use this class, you use the methods @p Functional::Add****Integral(integrand,domain_of_integration)
 * where @p integrand is a q-function lambda or functor and @p domain_of_integration is an @p mfem::mesh
 *
 * @see https://libceed.readthedocs.io/en/latest/libCEEDapi/#theoretical-framework for additional
 * information on the idea behind a quadrature function and its inputs/outputs
 *
 * @code{.cpp}
 * // for domains made up of quadrilaterals embedded in R^2
 * my_residual.AddAreaIntegral(integrand, domain_of_integration);
 * // alternatively...
 * my_residual.AddDomainIntegral(Dimension<2>{}, integrand, domain_of_integration);
 *
 * // for domains made up of quadrilaterals embedded in R^3
 * my_residual.AddSurfaceIntegral(integrand, domain_of_integration);
 *
 * // for domains made up of hexahedra embedded in R^3
 * my_residual.AddVolumeIntegral(integrand, domain_of_integration);
 * // alternatively...
 * my_residual.AddDomainIntegral(Dimension<3>{}, integrand, domain_of_integration);
 * @endcode
 */
template <typename test, typename... trials, ExecutionSpace exec>
class Functional<test(trials...), exec> {
  static constexpr camp::tuple<trials...> trial_spaces{};
  static constexpr uint32_t               num_trial_spaces = sizeof...(trials);

  class Gradient;

  // clang-format off
  template <int i> 
  struct operator_paren_return {
    using type = typename std::conditional<
        i >= 0,                                 // if `i` is greater than or equal to zero,
        camp::tuple<mfem::Vector&, Gradient&>, // then we return the value and the derivative w.r.t arg `i`
        mfem::Vector&                           // otherwise, we just return the value
        >::type;
  };
  // clang-format on

public:
  /**
   * @brief Constructs using @p mfem::ParFiniteElementSpace objects corresponding to the test/trial spaces
   * @param[in] test_fes The (non-qoi) test space
   * @param[in] trial_fes The trial space
   */
  Functional(mfem::ParFiniteElementSpace*                               test_fes,
             std::array<mfem::ParFiniteElementSpace*, num_trial_spaces> trial_fes)
      : test_space_(test_fes), trial_space_(trial_fes)
  {
    for (uint32_t i = 0; i < num_trial_spaces; i++) {
      P_trial_[i] = trial_space_[i]->GetProlongationMatrix();
      G_trial_[i] = trial_space_[i]->GetElementRestriction(mfem::ElementDofOrdering::LEXICOGRAPHIC);
      SLIC_ERROR_IF(!G_trial_[i], "Couldn't retrieve element restriction operator for trial space");

      if (compatibleWithFaceRestriction(*trial_space_[i])) {
        G_trial_boundary_[i] = trial_space_[i]->GetFaceRestriction(
            mfem::ElementDofOrdering::LEXICOGRAPHIC, mfem::FaceType::Boundary, mfem::L2FaceValues::SingleValued);

        input_E_boundary_[i].SetSize(G_trial_boundary_[i]->Height(), mfem::Device::GetMemoryType());
      }

      input_L_[i].SetSize(P_trial_[i]->Height(), mfem::Device::GetMemoryType());
      input_E_[i].SetSize(G_trial_[i]->Height(), mfem::Device::GetMemoryType());

      // create the gradient operators for each trial space
      grad_.emplace_back(*this, i);
    }

    P_test_ = test_space_->GetProlongationMatrix();
    G_test_ = test_space_->GetElementRestriction(mfem::ElementDofOrdering::LEXICOGRAPHIC);
    SLIC_ERROR_IF(!G_test_, "Couldn't retrieve element restriction operator for test space");

    // for now, limitations in mfem prevent us from implementing surface integrals for Hcurl test/trial space
    if (compatibleWithFaceRestriction(*test_space_)) {
      G_test_boundary_ = test_space_->GetFaceRestriction(mfem::ElementDofOrdering::LEXICOGRAPHIC,
                                                         mfem::FaceType::Boundary, mfem::L2FaceValues::SingleValued);
      output_E_boundary_.SetSize(G_test_boundary_->Height(), mfem::Device::GetMemoryType());
    }

    output_E_.SetSize(G_test_->Height(), mfem::Device::GetMemoryType());

    output_L_boundary_.SetSize(P_test_->Height(), mfem::Device::GetMemoryType());

    output_L_.SetSize(P_test_->Height(), mfem::Device::GetMemoryType());

    output_T_.SetSize(test_fes->GetTrueVSize(), mfem::Device::GetMemoryType());

    auto num_elements          = static_cast<size_t>(test_space_->GetNE());
    auto ndof_per_test_element = static_cast<size_t>(test_space_->GetFE(0)->GetDof() * test_space_->GetVDim());
    for (uint32_t i = 0; i < num_trial_spaces; i++) {
      auto ndof_per_trial_element =
          static_cast<size_t>(trial_space_[i]->GetFE(0)->GetDof() * trial_space_[i]->GetVDim());
      element_gradients_[i] = ExecArray<double, 3, exec>(num_elements, ndof_per_test_element, ndof_per_trial_element);
      bdr_element_gradients_[i] = allocateMemoryForBdrElementGradients<double, exec>(*test_space_, *trial_space_[i]);
    }
  }

  /**
   * @brief Adds a domain integral term to the weak formulation of the PDE
   * @tparam dim The dimension of the element (2 for quad, 3 for hex, etc)
   * @tparam lambda the type of the integrand functor: must implement operator() with an appropriate function signature
   * @tparam qpt_data_type The type of the data to store for each quadrature point
   * @param[in] integrand The user-provided quadrature function, see @p Integral
   * @param[in] domain The domain on which to evaluate the integral
   * @note The @p Dimension parameters are used to assist in the deduction of the @a geometry_dim
   * and @a spatial_dim template parameter
   * @param[inout] qdata The data for each quadrature point
   */
  template <int dim, typename lambda, typename qpt_data_type = Nothing>
  void AddDomainIntegral(Dimension<dim>, lambda&& integrand, mfem::Mesh& domain,
                         std::shared_ptr<QuadratureData<qpt_data_type>> qdata = NoQData)
  {
    auto num_elements = domain.GetNE();
    if (num_elements == 0) return;

    SLIC_ERROR_ROOT_IF(dim != domain.Dimension(), "invalid mesh dimension for domain integral");
    for (int e = 0; e < num_elements; e++) {
      SLIC_ERROR_ROOT_IF(domain.GetElementType(e) != supported_types[dim], "Mesh contains unsupported element type");
    }

    const mfem::FiniteElement&   el = *test_space_->GetFE(0);
    const mfem::IntegrationRule& ir = mfem::IntRules.Get(el.GetGeomType(), el.GetOrder() * 2);

    constexpr auto flags = mfem::GeometricFactors::COORDINATES | mfem::GeometricFactors::JACOBIANS;

    // NOTE: we are relying on MFEM to keep these geometric factors accurate. We store
    // the necessary data as references in the integral data structure.
    auto geom = domain.GetGeometricFactors(ir, flags);
    domain_integrals_.emplace_back(num_elements, geom->J, geom->X, Dimension<dim>{}, integrand, qdata);
  }

  /**
   * @brief Adds a boundary integral term to the weak formulation of the PDE
   * @tparam dim The dimension of the boundary element (1 for line, 2 for quad, etc)
   * @tparam lambda the type of the integrand functor: must implement operator() with an appropriate function signature
   * @param[in] integrand The user-provided quadrature function, see @p Integral
   * @param[in] domain The domain on which to evaluate the integral
   * @note The @p Dimension parameters are used to assist in the deduction of the @a geometry_dim
   * and @a spatial_dim template parameter
   */
  template <int dim, typename lambda>
  void AddBoundaryIntegral(Dimension<dim>, lambda&& integrand, mfem::Mesh& domain)
  {
    // TODO: fix mfem::FaceGeometricFactors
    auto num_bdr_elements = domain.GetNBE();
    if (num_bdr_elements == 0) return;

    SLIC_ERROR_ROOT_IF((dim + 1) != domain.Dimension(), "invalid mesh dimension for boundary integral");
    for (int e = 0; e < num_bdr_elements; e++) {
      SLIC_ERROR_ROOT_IF(domain.GetBdrElementType(e) != supported_types[dim], "Mesh contains unsupported element type");
    }

    const mfem::FiniteElement&   el = *test_space_->GetFE(0);
    const mfem::IntegrationRule& ir = mfem::IntRules.Get(supported_types[dim], el.GetOrder() * 2);
    constexpr auto flags = mfem::FaceGeometricFactors::COORDINATES | mfem::FaceGeometricFactors::DETERMINANTS |
                           mfem::FaceGeometricFactors::NORMALS;

    // NOTE: we are relying on MFEM to keep these geometric factors accurate. We store
    // the necessary data as references in the integral data structure.

    // Despite what their documentation says, mfem doesn't actually support the JACOBIANS flag.
    // this is currently a dealbreaker, as we need this information to do any calculations
    auto geom = domain.GetFaceGeometricFactors(ir, flags, mfem::FaceType::Boundary);

    bdr_integrals_.emplace_back(num_bdr_elements, geom->detJ, geom->X, geom->normal, Dimension<dim>{}, integrand);
  }

  /**
   * @brief Adds an area integral, i.e., over 2D elements in R^2 space
   * @tparam lambda the type of the integrand functor: must implement operator() with an appropriate function signature
   * @tparam qpt_data_type The type of the data to store for each quadrature point
   * @param[in] integrand The quadrature function
   * @param[in] domain The mesh to evaluate the integral on
   * @param[inout] data The data for each quadrature point
   */
  template <typename lambda, typename qpt_data_type = Nothing>
  void AddAreaIntegral(lambda&& integrand, mfem::Mesh& domain,
                       std::shared_ptr<QuadratureData<qpt_data_type>> data = NoQData)
  {
    AddDomainIntegral(Dimension<2>{}, integrand, domain, data);
  }

  /**
   * @brief Adds a volume integral, i.e., over 3D elements in R^3 space
   * @tparam lambda the type of the integrand functor: must implement operator() with an appropriate function signature
   * @tparam qpt_data_type The type of the data to store for each quadrature point
   * @param[in] integrand The quadrature function
   * @param[in] domain The mesh to evaluate the integral on
   * @param[inout] data The data for each quadrature point
   */
  template <typename lambda, typename qpt_data_type = Nothing>
  void AddVolumeIntegral(lambda&& integrand, mfem::Mesh& domain,
                         std::shared_ptr<QuadratureData<qpt_data_type>> data = NoQData)
  {
    AddDomainIntegral(Dimension<3>{}, integrand, domain, data);
  }

  /// @brief alias for Functional::AddBoundaryIntegral(Dimension<2>{}, integrand, domain);
  template <typename lambda>
  void AddSurfaceIntegral(lambda&& integrand, mfem::Mesh& domain)
  {
    AddBoundaryIntegral(Dimension<2>{}, integrand, domain);
  }

  /**
   * @brief this function computes the directional derivative of `serac::Functional::operator()`
   *
   * @param input_T the T-vector to apply the action of gradient to
   * @param output_T the T-vector where the resulting values are stored
   * @param which describes which trial space input_T corresponds to
   *
   * @note: it accepts exactly `num_trial_spaces` arguments of type mfem::Vector. Additionally, one of those
   * arguments may be a dual_vector, to indicate that Functional::operator() should not only evaluate the
   * element calculations, but also differentiate them w.r.t. the specified dual_vector argument
   */
  void ActionOfGradient(const mfem::Vector& input_T, mfem::Vector& output_T, std::size_t which) const
  {
    P_trial_[which]->Mult(input_T, input_L_[which]);

    output_L_ = 0.0;
    if (domain_integrals_.size() > 0) {
      // get the values for each element on the local processor
      G_trial_[which]->Mult(input_L_[which], input_E_[which]);

      // compute residual contributions at the element level and sum them

      output_E_ = 0.0;
      for (auto& integral : domain_integrals_) {
        integral.GradientMult(input_E_[which], output_E_, which);
      }

      // scatter-add to compute residuals on the local processor
      G_test_->MultTranspose(output_E_, output_L_);
    }

    if (bdr_integrals_.size() > 0) {
      G_trial_boundary_[which]->Mult(input_L_[which], input_E_boundary_[which]);

      output_E_boundary_ = 0.0;
      for (auto& integral : bdr_integrals_) {
        integral.GradientMult(input_E_boundary_[which], output_E_boundary_, which);
      }

      output_L_boundary_ = 0.0;

      // scatter-add to compute residuals on the local processor
      G_test_boundary_->MultTranspose(output_E_boundary_, output_L_boundary_);

      output_L_ += output_L_boundary_;
    }

    // scatter-add to compute global residuals
    P_test_->MultTranspose(output_L_, output_T);
  }

  /**
   * @brief this function lets the user evaluate the serac::Functional with the given trial space values
   *
   * note: it accepts exactly `num_trial_spaces` arguments of type mfem::Vector. Additionally, one of those
   * arguments may be a dual_vector, to indicate that Functional::operator() should not only evaluate the
   * element calculations, but also differentiate them w.r.t. the specified dual_vector argument
   *
   * @tparam T the types of the arguments passed in
   * @param args the trial space dofs used to carry out the calculation,
   *  at most one of which may be of the type `differentiate_wrt_this(mfem::Vector)`
   */
  template <int wrt, typename... T>
  typename operator_paren_return<wrt>::type operator()(DifferentiateWRT<wrt>, const T&... args)
  {
    const mfem::Vector* input_T[] = {&static_cast<const mfem::Vector&>(args)...};

    // get the values for each local processor
    for (uint32_t i = 0; i < num_trial_spaces; i++) {
      P_trial_[i]->Mult(*input_T[i], input_L_[i]);
    }

    output_L_ = 0.0;
    if (domain_integrals_.size() > 0) {
      // get the values for each element on the local processor
      for (uint32_t i = 0; i < num_trial_spaces; i++) {
        G_trial_[i]->Mult(input_L_[i], input_E_[i]);
      }

      // compute residual contributions at the element level and sum them
      output_E_ = 0.0;
      for (auto& integral : domain_integrals_) {
        integral.Mult(input_E_, output_E_, wrt, update_qdata);
      }

      // scatter-add to compute residuals on the local processor
      G_test_->MultTranspose(output_E_, output_L_);
    }

    if (bdr_integrals_.size() > 0) {
      for (uint32_t i = 0; i < num_trial_spaces; i++) {
        G_trial_boundary_[i]->Mult(input_L_[i], input_E_boundary_[i]);
      }

      output_E_boundary_ = 0.0;
      for (auto& integral : bdr_integrals_) {
        integral.Mult(input_E_boundary_, output_E_boundary_, wrt);
      }

      output_L_boundary_ = 0.0;

      // scatter-add to compute residuals on the local processor
      G_test_boundary_->MultTranspose(output_E_boundary_, output_L_boundary_);

      output_L_ += output_L_boundary_;
    }

    // scatter-add to compute global residuals
    P_test_->MultTranspose(output_L_, output_T_);

    if constexpr (wrt >= 0) {
      // if the user has indicated they'd like to evaluate and differentiate w.r.t.
      // a specific argument, then we return both the value and gradient w.r.t. that argument
      //
      // mfem::Vector arg0 = ...;
      // mfem::Vector arg1 = ...;
      // e.g. auto [value, gradient_wrt_arg1] = my_functional(arg0, differentiate_wrt(arg1));
      return typename operator_paren_return_index<wrt>::type{output_T_, grad_[wrt]};
    }
    if constexpr (wrt == -1) {
      // if the user passes only `mfem::Vector`s then we assume they only want the output value
      //
      // mfem::Vector arg0 = ...;
      // mfem::Vector arg1 = ...;
      // e.g. mfem::Vector value = my_functional(arg0, arg1);
      return output_T_;
    }
  }

  /// @overload
  template <typename... T>
  auto operator()(const T&... args)
  {
    constexpr int num_differentiated_arguments = (std::is_same_v<T, differentiate_wrt_this> + ...);
    static_assert(num_differentiated_arguments <= 1,
                  "Error: Functional::operator() can only differentiate w.r.t. 1 argument a time");
    static_assert(sizeof...(T) == num_trial_spaces,
                  "Error: Functional::operator() must take exactly as many arguments as trial spaces");

    [[maybe_unused]] constexpr int i = index_of_differentiation<T...>();

    return (*this)(DifferentiateWRT<i>{}, args...);
  }

  // TODO: expose this feature a better way
  /// @brief flag for denoting when a residual evaluation should update the material state buffers
  bool update_qdata;

private:
  /**
   * @brief mfem::Operator representing the gradient matrix that
   * can compute the action of the gradient (with operator()),
   * or assemble the sparse matrix representation through implicit conversion to mfem::HypreParMatrix *
   */
  class Gradient : public mfem::Operator {
  public:
    /**
     * @brief Constructs a Gradient wrapper that references a parent @p Functional
     * @param[in] f The @p Functional to use for gradient calculations
     */
    Gradient(Functional<test(trials...), exec>& f, uint32_t which = 0)
        : mfem::Operator(f.test_space_->GetTrueVSize(), f.trial_space_[which]->GetTrueVSize()),
          form_(f),
          lookup_tables(*(f.test_space_), *(f.trial_space_[which])),
          which_argument(which),
          test_space_(f.test_space_),
          trial_space_(f.trial_space_[which]),
          df_(f.test_space_->GetTrueVSize())
    {
    }

    /**
     * @brief implement that action of the gradient: df := df_dx * dx
     * @param[in] dx a small perturbation in the trial space
     * @param[in] df the resulting small perturbation in the residuals
     */
    virtual void Mult(const mfem::Vector& dx, mfem::Vector& df) const override
    {
      form_.ActionOfGradient(dx, df, which_argument);
    }

    /// @brief syntactic sugar:  df_dx.Mult(dx, df)  <=>  mfem::Vector df = df_dx(dx);
    mfem::Vector& operator()(const mfem::Vector& dx)
    {
      form_.ActionOfGradient(dx, df_, which_argument);
      return df_;
    }

    /// @brief assemble element matrices and form an mfem::HypreParMatrix
    std::unique_ptr<mfem::HypreParMatrix> assemble()
    {
      // the CSR graph (sparsity pattern) is reusable, so we cache
      // that and ask mfem to not free that memory in ~SparseMatrix()
      constexpr bool sparse_matrix_frees_graph_ptrs = false;

      // the CSR values are NOT reusable, so we pass ownership of
      // them to the mfem::SparseMatrix, to be freed in ~SparseMatrix()
      constexpr bool sparse_matrix_frees_values_ptr = true;

      constexpr bool col_ind_is_sorted = true;

      double* values = new double[lookup_tables.nnz]{};

      // each element uses the lookup tables to add its contributions
      // to their appropriate locations in the global sparse matrix
      if (form_.domain_integrals_.size() > 0) {
        auto& K_elem = form_.element_gradients_[which_argument];
        auto& LUT    = lookup_tables.element_nonzero_LUT;

        detail::zero_out(K_elem);
        for (auto& domain : form_.domain_integrals_) {
          domain.ComputeElementGradients(view(K_elem), which_argument);
        }

        for (axom::IndexType e = 0; e < K_elem.shape()[0]; e++) {
          for (axom::IndexType i = 0; i < K_elem.shape()[1]; i++) {
            for (axom::IndexType j = 0; j < K_elem.shape()[2]; j++) {
              auto [index, sign] = LUT(e, i, j);
              values[index] += sign * K_elem(e, i, j);
            }
          }
        }
      }

      // each boundary element uses the lookup tables to add its contributions
      // to their appropriate locations in the global sparse matrix
      if (form_.bdr_integrals_.size() > 0) {
        auto& K_belem = form_.bdr_element_gradients_[which_argument];
        auto& LUT     = lookup_tables.bdr_element_nonzero_LUT;

        detail::zero_out(K_belem);
        for (auto& boundary : form_.bdr_integrals_) {
          boundary.ComputeElementGradients(view(K_belem), which_argument);
        }

        for (axom::IndexType e = 0; e < K_belem.shape()[0]; e++) {
          for (axom::IndexType i = 0; i < K_belem.shape()[1]; i++) {
            for (axom::IndexType j = 0; j < K_belem.shape()[2]; j++) {
              auto [index, sign] = LUT(e, i, j);
              values[index] += sign * K_belem(e, i, j);
            }
          }
        }
      }

      // Copy the column indices to an auxilliary array as MFEM can mutate these during HypreParMatrix construction
      col_ind_copy_ = lookup_tables.col_ind;

      auto J_local =
          mfem::SparseMatrix(lookup_tables.row_ptr.data(), col_ind_copy_.data(), values, form_.output_L_.Size(),
                             form_.input_L_[which_argument].Size(), sparse_matrix_frees_graph_ptrs,
                             sparse_matrix_frees_values_ptr, col_ind_is_sorted);

      auto* R = form_.test_space_->Dof_TrueDof_Matrix();

      auto* A =
          new mfem::HypreParMatrix(test_space_->GetComm(), test_space_->GlobalVSize(), trial_space_->GlobalVSize(),
                                   test_space_->GetDofOffsets(), trial_space_->GetDofOffsets(), &J_local);

      auto* P = trial_space_->Dof_TrueDof_Matrix();

      std::unique_ptr<mfem::HypreParMatrix> K(mfem::RAP(R, A, P));

      delete A;

      return K;
    };

    friend auto assemble(Gradient& g) { return g.assemble(); }

  private:
    /// @brief The "parent" @p Functional to calculate gradients with
    Functional<test(trials...), exec>& form_;

    /**
     * @brief this object has lookup tables for where to place each
     *   element and boundary element gradient contribution in the global
     *   sparse matrix
     */
    GradientAssemblyLookupTables lookup_tables;

    /**
     * @brief Copy of the column indices for sparse matrix assembly
     * @note These are mutated by MFEM during HypreParMatrix construction
     */
    std::vector<int> col_ind_copy_;

    /**
     * @brief this member variable tells us which argument the associated Functional this gradient
     *  corresponds to:
     *
     *  e.g.
     *    Functional< test(trial0, trial1, trial2) > f(...);
     *    grad<0>(f) == df_dtrial0
     *    grad<1>(f) == df_dtrial1
     *    grad<2>(f) == df_dtrial2
     */
    uint32_t which_argument;

    /// @brief shallow copy of the test space from the associated Functional
    mfem::ParFiniteElementSpace* test_space_;

    /// @brief shallow copy of the trial space from the associated Functional
    mfem::ParFiniteElementSpace* trial_space_;

    /// @brief storage for computing the action-of-gradient output
    mfem::Vector df_;
  };

  /// @brief The input set of local DOF values (i.e., on the current rank)
  mutable mfem::Vector input_L_[num_trial_spaces];

  /// @brief The output set of local DOF values (i.e., on the current rank)
  mutable mfem::Vector output_L_;

  /// @brief The input set of per-element DOF values
  mutable std::array<mfem::Vector, num_trial_spaces> input_E_;

  /// @brief The output set of per-element DOF values
  mutable mfem::Vector output_E_;

  /// @brief The input set of per-boundaryelement DOF values
  mutable std::array<mfem::Vector, num_trial_spaces> input_E_boundary_;

  /// @brief The output set of per-boundary-element DOF values
  mutable mfem::Vector output_E_boundary_;

  /// @brief The output set of local DOF values (i.e., on the current rank) from boundary elements
  mutable mfem::Vector output_L_boundary_;

  /// @brief The set of true DOF values, a reference to this member is returned by @p operator()
  mutable mfem::Vector output_T_;

  /// @brief Manages DOFs for the test space
  mfem::ParFiniteElementSpace* test_space_;

  /// @brief Manages DOFs for the trial space
  std::array<mfem::ParFiniteElementSpace*, num_trial_spaces> trial_space_;

  /**
   * @brief Operator that converts true (global) DOF values to local (current rank) DOF values
   * for the test space
   */
  const mfem::Operator* P_test_;

  /**
   * @brief Operator that converts local (current rank) DOF values to per-element DOF values
   * for the test space
   */
  const mfem::Operator* G_test_;

  /**
   * @brief Operator that converts true (global) DOF values to local (current rank) DOF values
   * for the trial space
   */
  const mfem::Operator* P_trial_[num_trial_spaces];

  /**
   * @brief Operator that converts local (current rank) DOF values to per-element DOF values
   * for the trial space
   */
  const mfem::Operator* G_trial_[num_trial_spaces];

  /**
   * @brief Operator that converts local (current rank) DOF values to per-boundary element DOF values
   * for the test space
   */
  const mfem::Operator* G_test_boundary_;

  /**
   * @brief Operator that converts local (current rank) DOF values to per-boundary element DOF values
   * for the trial space
   */
  const mfem::Operator* G_trial_boundary_[num_trial_spaces];

  /// @brief The set of domain integrals (spatial_dim == geometric_dim)
  std::vector<DomainIntegral<test(trials...), exec>> domain_integrals_;

  /// @brief The set of boundary integral (spatial_dim == geometric_dim + 1)
  std::vector<BoundaryIntegral<test(trials...), exec>> bdr_integrals_;

  // simplex elements are currently not supported;
  static constexpr mfem::Element::Type supported_types[4] = {mfem::Element::POINT, mfem::Element::SEGMENT,
                                                             mfem::Element::QUADRILATERAL, mfem::Element::HEXAHEDRON};

  /// @brief The objects representing the gradients w.r.t. each input argument of the Functional
  mutable std::vector<Gradient> grad_;

  /// @brief 3D array that stores each element's gradient of the residual w.r.t. trial values
  ExecArray<double, 3, exec> element_gradients_[num_trial_spaces];

  /// @brief 3D array that stores each boundary element's gradient of the residual w.r.t. trial values
  ExecArray<double, 3, exec> bdr_element_gradients_[num_trial_spaces];
};

}  // namespace serac

#include "functional_qoi.inl"
