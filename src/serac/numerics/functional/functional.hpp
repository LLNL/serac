// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
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

/**
 * @brief for reasons I don't understand,
 * these calls need to be made immediately after creating the mesh,
 * in order for mfem::FaceRestriction to work properly (?)
 *
 * @note Apparently, calling these functions also messes up Sidre, causing it to segfault, so..
 */
void make_the_mesh_work(mfem::ParMesh* mesh)
{
  mesh->EnsureNodes();
  mesh->ExchangeFaceNbrData();
}

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
template <typename test, typename trial, ExecutionSpace exec>
class Functional<test(trial), exec> : public mfem::Operator {
public:
  /**
   * @brief Constructs using @p mfem::ParFiniteElementSpace objects corresponding to the test/trial spaces
   * @param[in] test_fes The (non-qoi) test space
   * @param[in] trial_fes The trial space
   */
  Functional(mfem::ParFiniteElementSpace* test_fes, mfem::ParFiniteElementSpace* trial_fes)
      : Operator(test_fes->GetTrueVSize(), trial_fes->GetTrueVSize()),
        test_space_(test_fes),
        trial_space_(trial_fes),
        P_test_(test_space_->GetProlongationMatrix()),
        G_test_(test_space_->GetElementRestriction(mfem::ElementDofOrdering::LEXICOGRAPHIC)),
        P_trial_(trial_space_->GetProlongationMatrix()),
        G_trial_(trial_space_->GetElementRestriction(mfem::ElementDofOrdering::LEXICOGRAPHIC)),
        grad_(*this)
  {
    SLIC_ERROR_IF(!G_test_, "Couldn't retrieve element restriction operator for test space");
    SLIC_ERROR_IF(!G_trial_, "Couldn't retrieve element restriction operator for trial space");

    // Ensure the mesh has the appropriate neighbor information before constructing the face restriction operators
    if (test_space_) {
      test_space_->ExchangeFaceNbrData();
    }
    if (trial_space_) {
      trial_space_->ExchangeFaceNbrData();
    }

    // for now, limitations in mfem prevent us from implementing surface integrals for Hcurl test/trial space
    if (trial::family != Family::HCURL && test::family != Family::HCURL) {
      if (test_space_) {
        G_test_boundary_ = test_space_->GetFaceRestriction(mfem::ElementDofOrdering::LEXICOGRAPHIC,
                                                           mfem::FaceType::Boundary, mfem::L2FaceValues::SingleValued);
      }
      if (trial_space_) {
        G_trial_boundary_ = trial_space_->GetFaceRestriction(
            mfem::ElementDofOrdering::LEXICOGRAPHIC, mfem::FaceType::Boundary, mfem::L2FaceValues::SingleValued);
      }
      input_E_boundary_.SetSize(G_trial_boundary_->Height(), mfem::Device::GetMemoryType());
      output_E_boundary_.SetSize(G_test_boundary_->Height(), mfem::Device::GetMemoryType());
      output_L_boundary_.SetSize(P_test_->Height(), mfem::Device::GetMemoryType());
    }

    input_L_.SetSize(P_trial_->Height(), mfem::Device::GetMemoryType());
    input_E_.SetSize(G_trial_->Height(), mfem::Device::GetMemoryType());
    output_E_.SetSize(G_test_->Height(), mfem::Device::GetMemoryType());
    output_L_.SetSize(P_test_->Height(), mfem::Device::GetMemoryType());
    my_output_T_.SetSize(Height(), mfem::Device::GetMemoryType());
    dummy_.SetSize(Width(), mfem::Device::GetMemoryType());

    {
      auto num_elements           = static_cast<size_t>(test_space_->GetNE());
      auto ndof_per_test_element  = static_cast<size_t>(test_space_->GetFE(0)->GetDof() * test_space_->GetVDim());
      auto ndof_per_trial_element = static_cast<size_t>(trial_space_->GetFE(0)->GetDof() * trial_space_->GetVDim());
      element_gradients_          = axom::Array<double, 3, detail::execution_to_memory_v<exec>>(
          num_elements, ndof_per_test_element, ndof_per_trial_element);
    }

    {
      bdr_element_gradients_ = allocateMemoryForBdrElementGradients<double, exec>(*test_space_, *trial_space_);
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
   * @param[inout] data The data for each quadrature point
   */
  template <int dim, typename lambda, typename qpt_data_type = void>
  void AddDomainIntegral(Dimension<dim>, lambda&& integrand, mfem::Mesh& domain,
                         QuadratureData<qpt_data_type>& data = dummy_qdata)
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
    auto           geom  = domain.GetGeometricFactors(ir, flags);
    domain_integrals_.emplace_back(num_elements, geom->J, geom->X, Dimension<dim>{}, integrand, data);
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

    // despite what their documentation says, mfem doesn't actually support the JACOBIANS flag.
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
  template <typename lambda, typename qpt_data_type = void>
  void AddAreaIntegral(lambda&& integrand, mfem::Mesh& domain, QuadratureData<qpt_data_type>& data = dummy_qdata)
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
  template <typename lambda, typename qpt_data_type = void>
  void AddVolumeIntegral(lambda&& integrand, mfem::Mesh& domain, QuadratureData<qpt_data_type>& data = dummy_qdata)
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
   * @brief Implements mfem::Operator::Mult
   * @param[in] input_T The input vector
   * @param[out] output_T The output vector
   */
  void Mult(const mfem::Vector& input_T, mfem::Vector& output_T) const override
  {
    Evaluation<Operation::Mult>(input_T, output_T);
  }

  /**
   * @brief Implements mfem::Operator::GetGradient
   * @param[in] x The input vector where the gradient is evaluated
   *
   * Note: at present, this Functional::Gradient object only supports the action of the gradient (i.e. directional
   * derivative) We are looking into making that Functional::Gradient also be convertible to a sparse matrix format as
   * well.
   */
  mfem::Operator& GetGradient(const mfem::Vector& x) const override
  {
    Mult(x, dummy_);  // this is ugly
    return grad_;
  }

  /**
   * @brief Alias for @p Mult that uses a return value instead of an output parameter
   * @param[in] input_T The input vector
   */
  mfem::Vector& operator()(const mfem::Vector& input_T) const
  {
    Evaluation<Operation::Mult>(input_T, my_output_T_);
    return my_output_T_;
  }

  /**
   * @brief Obtains the gradients for all the constituent integrals
   * @param[in] input_T The input vector
   * @param[out] output_T The output vector
   * @see DomainIntegral::GradientMult, BoundaryIntegral::GradientMult
   */
  virtual void GradientMult(const mfem::Vector& input_T, mfem::Vector& output_T) const
  {
    Evaluation<Operation::GradientMult>(input_T, output_T);
  }

  /**
   * @brief Applies an essential boundary condition to the attributes specified by @a ess_attr
   * @param[in] ess_attr The mesh attributes to apply the BC to
   *
   * @note This gets more interesting when having more than one trial space
   */
  void SetEssentialBC(const mfem::Array<int>& ess_attr)
  {
    static_assert(std::is_same_v<test, trial>, "can't specify essential bc on incompatible spaces");
    trial_space_->GetEssentialTrueDofs(ess_attr, ess_tdof_list_);
  }

private:
  /**
   * @brief Indicates whether to obtain values or gradients from a calculation
   */
  enum class Operation
  {
    Mult,
    GradientMult
  };

  /**
   * @brief Lightweight shim for mfem::Operator that produces the gradient of a @p Functional from a @p Mult
   */
  class Gradient : public mfem::Operator {
  public:
    /**
     * @brief Constructs a Gradient wrapper that references a parent @p Functional
     * @param[in] f The @p Functional to use for gradient calculations
     */
    Gradient(Functional<test(trial), exec>& f)
        : mfem::Operator(f.Height(), f.Width()), form_(f), lookup_tables(*(f.test_space_), *(f.trial_space_)){};

    /**
     * @brief implement that action of the gradient: df := df_dx * dx
     * @param[in] dx a small perturbation in the trial space
     * @param[in] df the resulting small perturbation in the residuals
     */
    virtual void Mult(const mfem::Vector& dx, mfem::Vector& df) const override { form_.GradientMult(dx, df); }

    /**
     * @brief syntactic sugar:  df_dx.Mult(dx, df)  <=>  mfem::Vector df = df_dx(dx);
     */
    mfem::Vector operator()(const mfem::Vector& x) const
    {
      mfem::Vector y(form_.Height());
      form_.GradientMult(x, y);
      return y;
    }

    /**
     * @brief implicit conversion to mfem::HypreParMatrix type
     */
    operator mfem::HypreParMatrix *()
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
        auto& K_elem = form_.element_gradients_;
        auto& LUT    = lookup_tables.element_nonzero_LUT;

        detail::zero_out(K_elem);
        for (auto& domain : form_.domain_integrals_) {
          domain.ComputeElementGradients(K_elem);
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
        auto& K_belem = form_.bdr_element_gradients_;
        auto& LUT     = lookup_tables.bdr_element_nonzero_LUT;

        detail::zero_out(K_belem);
        for (auto& boundary : form_.bdr_integrals_) {
          boundary.ComputeElementGradients(K_belem);
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

      auto J_local = mfem::SparseMatrix(lookup_tables.row_ptr.data(), lookup_tables.col_ind.data(), values,
                                        form_.output_L_.Size(), form_.input_L_.Size(), sparse_matrix_frees_graph_ptrs,
                                        sparse_matrix_frees_values_ptr, col_ind_is_sorted);

      auto* R = form_.test_space_->Dof_TrueDof_Matrix();

      auto* A = new mfem::HypreParMatrix(form_.test_space_->GetComm(), form_.test_space_->GlobalVSize(),
                                         form_.trial_space_->GlobalVSize(), form_.test_space_->GetDofOffsets(),
                                         form_.trial_space_->GetDofOffsets(), &J_local);

      auto* P = form_.trial_space_->Dof_TrueDof_Matrix();

      auto* RAP = mfem::RAP(R, A, P);

      delete A;

      return RAP;
    }

  private:
    /// @brief The "parent" @p Functional to calculate gradients with
    Functional<test(trial), exec>& form_;

    /**
     * @brief this object has lookup tables for where to place each
     *   element and boundary element gradient contribution in the global
     *   sparse matrix
     */
    GradientAssemblyLookupTables lookup_tables;
  };

  /**
   * @brief Helper method for evaluation/gradient evaluation
   * @tparam op Whether to obtain values or gradients
   * @param[in] input_T The input vector
   * @param[out] output_T The output vector
   */
  template <Operation op = Operation::Mult>
  void Evaluation(const mfem::Vector& input_T, mfem::Vector& output_T) const
  {
    // get the values for each local processor
    P_trial_->Mult(input_T, input_L_);

    output_L_ = 0.0;
    if (domain_integrals_.size() > 0) {
      // get the values for each element on the local processor
      G_trial_->Mult(input_L_, input_E_);

      // compute residual contributions at the element level and sum them
      output_E_ = 0.0;
      for (auto& integral : domain_integrals_) {
        if constexpr (op == Operation::Mult) {
          integral.Mult(input_E_, output_E_);
        }

        if constexpr (op == Operation::GradientMult) {
          integral.GradientMult(input_E_, output_E_);
        }
      }

      // scatter-add to compute residuals on the local processor
      G_test_->MultTranspose(output_E_, output_L_);
    }

    if (bdr_integrals_.size() > 0) {
      G_trial_boundary_->Mult(input_L_, input_E_boundary_);

      output_E_boundary_ = 0.0;
      for (auto& integral : bdr_integrals_) {
        if constexpr (op == Operation::Mult) {
          integral.Mult(input_E_boundary_, output_E_boundary_);
        }

        if constexpr (op == Operation::GradientMult) {
          integral.GradientMult(input_E_boundary_, output_E_boundary_);
        }
      }

      output_L_boundary_ = 0.0;

      // scatter-add to compute residuals on the local processor
      G_test_boundary_->MultTranspose(output_E_boundary_, output_L_boundary_);

      output_L_ += output_L_boundary_;
    }

    // scatter-add to compute global residuals
    P_test_->MultTranspose(output_L_, output_T);

    output_T.HostReadWrite();
    for (int i = 0; i < ess_tdof_list_.Size(); i++) {
      if constexpr (op == Operation::Mult) {
        output_T(ess_tdof_list_[i]) = 0.0;
      }

      if constexpr (op == Operation::GradientMult) {
        output_T(ess_tdof_list_[i]) = input_T(ess_tdof_list_[i]);
      }
    }
  }

  /**
   * @brief The input set of local DOF values (i.e., on the current rank)
   */
  mutable mfem::Vector input_L_;

  /**
   * @brief The output set of local DOF values (i.e., on the current rank)
   */
  mutable mfem::Vector output_L_;

  /**
   * @brief The input set of per-element DOF values
   */
  mutable mfem::Vector input_E_;

  /**
   * @brief The output set of per-element DOF values
   */
  mutable mfem::Vector output_E_;

  /**
   * @brief The input set of per-boundaryelement DOF values
   */
  mutable mfem::Vector input_E_boundary_;

  /**
   * @brief The output set of per-boundary-element DOF values
   */
  mutable mfem::Vector output_E_boundary_;

  /**
   * @brief The output set of local DOF values (i.e., on the current rank) from boundary elements
   */
  mutable mfem::Vector output_L_boundary_;

  /**
   * @brief The set of true DOF values, used as a scratchpad for @p operator()
   */
  mutable mfem::Vector my_output_T_;

  /**
   * @brief A working vector for @p GetGradient
   */
  mutable mfem::Vector dummy_;

  /**
   * @brief Manages DOFs for the test space
   */
  mfem::ParFiniteElementSpace* test_space_;

  /**
   * @brief Manages DOFs for the trial space
   */
  mfem::ParFiniteElementSpace* trial_space_;

  /**
   * @brief The set of true DOF indices to which an essential BC should be applied
   */
  mfem::Array<int> ess_tdof_list_;

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
  const mfem::Operator* P_trial_;

  /**
   * @brief Operator that converts local (current rank) DOF values to per-element DOF values
   * for the trial space
   */
  const mfem::Operator* G_trial_;

  /**
   * @brief Operator that converts local (current rank) DOF values to per-boundary element DOF values
   * for the test space
   */
  const mfem::Operator* G_test_boundary_;

  /**
   * @brief Operator that converts local (current rank) DOF values to per-boundary element DOF values
   * for the trial space
   */
  const mfem::Operator* G_trial_boundary_;

  /**
   * @brief The set of domain integrals (spatial_dim == geometric_dim)
   */
  std::vector<DomainIntegral<test(trial), exec>> domain_integrals_;

  /**
   * @brief The set of boundary integral (spatial_dim > geometric_dim)
   */
  std::vector<BoundaryIntegral<test(trial), exec>> bdr_integrals_;

  // simplex elements are currently not supported;
  static constexpr mfem::Element::Type supported_types[4] = {mfem::Element::POINT, mfem::Element::SEGMENT,
                                                             mfem::Element::QUADRILATERAL, mfem::Element::HEXAHEDRON};

  /**
   * @brief The gradient object used to implement @p GetGradient
   */
  mutable Gradient grad_;

  /// @brief 3D array that stores each element's gradient of the residual w.r.t. trial values
  axom::Array<double, 3, detail::execution_to_memory_v<exec>> element_gradients_;

  /// @brief 3D array that stores each boundary element's gradient of the residual w.r.t. trial values
  axom::Array<double, 3, detail::execution_to_memory_v<exec>> bdr_element_gradients_;

  template <typename T>
  friend typename Functional<T>::Gradient& grad(Functional<T>&);
};

/**
 * @brief free function for accessing the gradient member of a Functional object
 *   intended to mimic the mathematical notation
 * @param[in] f the Functional whose gradient is returned
 */
template <typename T>
typename Functional<T>::Gradient& grad(Functional<T>& f)
{
  return f.grad_;
}

}  // namespace serac

#include "functional_qoi.inl"
