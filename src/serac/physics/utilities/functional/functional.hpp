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
#include "serac/physics/utilities/functional/tensor.hpp"
#include "serac/physics/utilities/functional/quadrature.hpp"
#include "serac/physics/utilities/functional/finite_element.hpp"
#include "serac/physics/utilities/functional/tuple_arithmetic.hpp"
#include "serac/physics/utilities/functional/domain_integral.hpp"
#include "serac/physics/utilities/functional/boundary_integral.hpp"
#include "serac/numerics/assembled_sparse_matrix.hpp"
#include "serac/infrastructure/logger.hpp"

namespace detail {

  struct elem_info{
    int global_row;
    int global_col;
    int local_row;
    int local_col;
    int element_id;
    int sign;
    bool on_boundary;
  };

  // for sorting lexicographically by {global_row, global_col}
  bool operator<(const elem_info & x, const elem_info & y) {
    return (x.global_row < y.global_row) || (x.global_row == y.global_row && x.global_col < y.global_col);
  }

  bool operator!=(const elem_info & x, const elem_info & y) {
    return (x.global_row != y.global_row) || (x.global_col != y.global_col);
  }

  int get_sign(int i) { return (i >= 0) ? 1 : -1; }
  int get_index(int i) { return (i >= 0) ? i : - 1 - i; }

  struct signed_index{
    int index;
    int sign;
    operator int(){ return index; }
  };

  void apply_permutation(mfem::Array<int> & input, const mfem::Array<int> & permutation) {
    auto output = input;
    for (int i = 0; i < permutation.Size(); i++) {
      if (permutation[i] >= 0) {
        output[i] = input[permutation[i]];
      } else {
        output[i] = -input[-permutation[i]-1]-1;
      }
    }
    input = output;
  }



}

namespace serac {

/// @cond
template <typename T, typename execution_policy = serac::default_policy>
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
 * @tparam execution_policy which kind of processor should be used to carry out calculations
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
template <typename test, typename trial, typename execution_policy>
class Functional<test(trial), execution_policy> : public mfem::Operator {

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
    if (test_space_) test_space_->ExchangeFaceNbrData();
    if (trial_space_) trial_space_->ExchangeFaceNbrData();

    // for now, limitations in mfem prevent us from implementing surface integrals for Hcurl test/trial space
    if (trial::family != Family::HCURL && test::family != Family::HCURL) {
      if (test_space_) {
        G_test_boundary_  = test_space_->GetFaceRestriction(mfem::ElementDofOrdering::LEXICOGRAPHIC, mfem::FaceType::Boundary, mfem::L2FaceValues::SingleValued);
      }
      if (trial_space_) {
        G_trial_boundary_  = trial_space_->GetFaceRestriction(mfem::ElementDofOrdering::LEXICOGRAPHIC, mfem::FaceType::Boundary, mfem::L2FaceValues::SingleValued);
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
  }

  /**
   * @brief Adds a domain integral term to the weak formulation of the PDE
   * @tparam dim The dimension of the element (2 for quad, 3 for hex, etc)
   * @tparam lambda the type of the integrand functor: must implement operator() with an appropriate function signature
   * @tparam qpt_data_type The type of the data to store for each quadrature point
   * @param[in] integrand The user-provided quadrature function, see @p Integral
   * @param[in] domain The domain on which to evaluate the integral
   * @param[in] data The data structure containing per-quadrature-point data
   * @note The @p Dimension parameters are used to assist in the deduction of the @a geometry_dim
   * and @a spatial_dim template parameter
   */
  template <int dim, typename lambda, typename qpt_data_type = void>
  void AddDomainIntegral(Dimension<dim>, lambda&& integrand, mfem::Mesh& domain)
  {
    auto num_elements = domain.GetNE();
    if (num_elements == 0) return;

    SLIC_ERROR_ROOT_IF(dim != domain.Dimension(), "Error: invalid mesh dimension for domain integral");
    for (int e = 0; e < num_elements; e++) {
      SLIC_ERROR_ROOT_IF(domain.GetElementType(e) != supported_types[dim], "Mesh contains unsupported element type");
    }

    const mfem::FiniteElement&   el = *test_space_->GetFE(0);
    const mfem::IntegrationRule& ir = mfem::IntRules.Get(el.GetGeomType(), el.GetOrder() * 2);

    constexpr auto flags = mfem::GeometricFactors::COORDINATES | mfem::GeometricFactors::JACOBIANS;
    auto           geom  = domain.GetGeometricFactors(ir, flags);
    domain_integrals_.emplace_back(num_elements, geom->J, geom->X, Dimension<dim>{}, integrand);
  }

  /**
   * @brief Adds a boundary integral term to the weak formulation of the PDE
   * @tparam dim The dimension of the boundary element (1 for line, 2 for quad, etc)
   * @tparam lambda the type of the integrand functor: must implement operator() with an appropriate function signature
   * @tparam qpt_data_type The type of the data to store for each quadrature point
   * @param[in] integrand The user-provided quadrature function, see @p Integral
   * @param[in] domain The domain on which to evaluate the integral
   * @param[in] data The data structure containing per-quadrature-point data
   * @note The @p Dimension parameters are used to assist in the deduction of the @a geometry_dim
   * and @a spatial_dim template parameter
   */
  template <int dim, typename lambda, typename qpt_data_type = void>
  void AddBoundaryIntegral(Dimension<dim>, lambda&& integrand, mfem::Mesh& domain)
  {
    // TODO: fix mfem::FaceGeometricFactors
    auto num_boundary_elements = domain.GetNBE();
    if (num_boundary_elements == 0) return;

    SLIC_ERROR_ROOT_IF((dim + 1) != domain.Dimension(), "Error: invalid mesh dimension for boundary integral");
    for (int e = 0; e < num_boundary_elements; e++) {
      SLIC_ERROR_ROOT_IF(domain.GetBdrElementType(e) != supported_types[dim], "Mesh contains unsupported element type");
    }

    const mfem::FiniteElement&   el = *test_space_->GetFE(0);
    const mfem::IntegrationRule& ir = mfem::IntRules.Get(supported_types[dim], el.GetOrder() * 2);
    constexpr auto flags = mfem::FaceGeometricFactors::COORDINATES | mfem::FaceGeometricFactors::DETERMINANTS |
                           mfem::FaceGeometricFactors::NORMALS;

    // despite what their documentation says, mfem doesn't actually support the JACOBIANS flag.
    // this is currently a dealbreaker, as we need this information to do any calculations
    auto geom = domain.GetFaceGeometricFactors(ir, flags, mfem::FaceType::Boundary);
    boundary_integrals_.emplace_back(num_boundary_elements, geom->detJ, geom->X, geom->normal, Dimension<dim>{},
                                     integrand);
  }

  /**
   * @brief Adds an area integral, i.e., over 2D elements in R^2 space
   * @tparam lambda the type of the integrand functor: must implement operator() with an appropriate function signature
   * @tparam qpt_data_type The type of the data to store for each quadrature point
   * @param[in] integrand The quadrature function
   * @param[in] domain The mesh to evaluate the integral on
   * @param[in] data The data structure containing per-quadrature-point data
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
   * @param[in] data The data structure containing per-quadrature-point data
   */
  template <typename lambda, typename qpt_data_type = void>
  void AddVolumeIntegral(lambda&& integrand, mfem::Mesh& domain, QuadratureData<qpt_data_type>& data = dummy_qdata)
  {
    AddDomainIntegral(Dimension<3>{}, integrand, domain, data);
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
  mfem::Vector & operator()(const mfem::Vector& input_T) const
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
   * @brief calculate and return the element stiffness matrices flattened into a mfem::Vector
   * @returns A mfem::Vector containing the element stiffness matrix entries (flattened from a 3D array 
   * with dimensions test_dim * test_ndof, trial_dim * trial_ndof, nelem)
   */
  mfem::Vector ComputeElementMatrices()
  {
    // Resize K_e_ if this is the first time
    if (K_e_.Size() == 0) {
      const auto& test_el  = *test_space_->GetFE(0);
      const auto& trial_el = *trial_space_->GetFE(0);
      K_e_.SetSize(test_el.GetDof() * test_space_->GetVDim() * trial_el.GetDof() * trial_space_->GetVDim() *
                   test_space_->GetNE());
    }
    // zero out internal vector
    K_e_ = 0.;
    // loop through integrals and accumulate
    for (auto & domain : domain_integrals_) domain.ComputeElementMatrices(K_e_);

    return K_e_;
  }

  /**
   * @brief calculate and return the boundary element stiffness matrices flattened into a mfem::Vector
   * @returns A mfem::Vector containing the boundary element matrix entries (flattened from a 3D array 
   * with dimensions test_dim * test_ndof, trial_dim * trial_ndof, nelem)
   */
  mfem::Vector ComputeBoundaryElementMatrices()
  {
    // Resize K_b_ if this is the first time
    if (K_b_.Size() == 0) {
      int num_boundary_elements = test_space_->GetNBE();
      int dofs_per_test_boundary_element = test_space_->GetBE(0)->GetDof() * test_space_->GetVDim();
      int dofs_per_trial_boundary_element = trial_space_->GetBE(0)->GetDof() * trial_space_->GetVDim();
      K_b_.SetSize(dofs_per_test_boundary_element * dofs_per_trial_boundary_element * num_boundary_elements);
    }
    // zero out internal vector
    K_b_ = 0.;
    // loop through integrals and accumulate
    for (auto & boundary : boundary_integrals_) boundary.ComputeElementMatrices(K_b_);

    return K_b_;
  }



  /**
   * @brief Computes element matrices and returns AssembledSparseMatrix
   * @return reference to AssembledSparseMatrix with newly assembled entries
   */

  serac::mfem_ext::AssembledSparseMatrix& GetAssembledSparseMatrix()
  {
    ComputeElementMatrices();  // Updates K_e_
    if (!assembled_spmat_) {
      assembled_spmat_ = std::make_unique<serac::mfem_ext::AssembledSparseMatrix>(
          *test_space_, *trial_space_, mfem::ElementDofOrdering::LEXICOGRAPHIC);
    }
    assembled_spmat_->FillData(K_e_);
    return *assembled_spmat_;
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

    template < typename T >
    struct Array3D{
      Array3D() = default;
      Array3D(int n1, int n2, int n3) : strides{n2 * n3, n3, 1}, data(n1 * n2 * n3) {}
      auto & operator()(int i, int j, int k) { return data[i * strides[0] + j * strides[1] + k * strides[2]]; }
      int strides[3];
      std::vector < T > data;
    };

  public:
    /**
     * @brief Constructs a Gradient wrapper that references a parent @p Functional
     * @param[in] f The @p Functional to use for gradient calculations
     */
    Gradient(Functional<test(trial)> & f) : mfem::Operator(f.Height(), f.Width()), form(f), sparsity_pattern_initialized(false) {};

    virtual void Mult(const mfem::Vector& x, mfem::Vector& y) const override { form.GradientMult(x, y); }

    void initialize_sparsity_pattern() {

      mfem::Array<int> test_dofs;
      mfem::Array<int> trial_dofs;

      int test_vdim  = form.test_space_->GetVDim();
      int trial_vdim = form.trial_space_->GetVDim();

      form.test_space_->GetElementDofs(0, test_dofs);
      form.trial_space_->GetElementDofs(0, trial_dofs);
      int num_elements = form.test_space_->GetNE();
      int dofs_per_test_element = test_dofs.Size();
      int dofs_per_trial_element = trial_dofs.Size();
      int entries_per_element = dofs_per_test_element * dofs_per_trial_element;

      form.test_space_->GetBdrElementDofs(0, test_dofs);
      form.trial_space_->GetBdrElementDofs(0, trial_dofs);
      int num_boundary_elements = form.test_space_->GetNBE();
      int dofs_per_test_boundary_element = test_dofs.Size();
      int dofs_per_trial_boundary_element = trial_dofs.Size();
      int entries_per_boundary_element = dofs_per_test_boundary_element * test_vdim * dofs_per_trial_boundary_element * trial_vdim;

      int num_infos[2] = {
        (form.domain_integrals_.size() > 0) * entries_per_element * num_elements,
        (form.boundary_integrals_.size() > 0) * entries_per_boundary_element * num_boundary_elements
      };

      std::vector < ::detail::elem_info > infos;
      infos.reserve(num_infos[0] + num_infos[1]);

      if (form.domain_integrals_.size() > 0) {
        bool on_boundary = false;

        for (int e = 0; e < num_elements; e++) {
          form.test_space_->GetElementDofs(e, test_dofs);
          form.trial_space_->GetElementDofs(e, trial_dofs);

          const mfem::Array<int> & test_native_to_lexicographic = dynamic_cast<const mfem::TensorBasisElement *>(form.test_space_->GetFE(0))->GetDofMap();
          const mfem::Array<int> & trial_native_to_lexicographic = dynamic_cast<const mfem::TensorBasisElement *>(form.trial_space_->GetFE(0))->GetDofMap();
          ::detail::apply_permutation(test_dofs, test_native_to_lexicographic);
          ::detail::apply_permutation(trial_dofs, trial_native_to_lexicographic);
          for (int i = 0; i < dofs_per_test_element; i++) {
            for (int j = 0; j < dofs_per_trial_element; j++) {
              for (int k = 0; k < test_vdim; k++) {
                int test_vdof = form.test_space_->DofToVDof(::detail::get_index(test_dofs[i]), k);
                for (int l = 0; l < trial_vdim; l++) {
                  int trial_vdof = form.trial_space_->DofToVDof(::detail::get_index(trial_dofs[j]), l);
                  infos.push_back({test_vdof, trial_vdof, i + dofs_per_test_element * k, j + dofs_per_trial_element * l, e, ::detail::get_sign(test_dofs[i]) * ::detail::get_sign(trial_dofs[j]), on_boundary});
                }
              }
            }
          }
        }

      }

      // mfem doesn't implement GetDofMap for some of its Nedelec elements (??),
      // so we have to temporarily disable boundary terms for Hcurl until they do
      if (form.boundary_integrals_.size() > 0) {
        bool on_boundary = true;

        for (int b = 0; b < num_boundary_elements; b++) {
          form.test_space_->GetBdrElementDofs(b, test_dofs);
          form.trial_space_->GetBdrElementDofs(b, trial_dofs);

          if constexpr (test::family != Family::HCURL) {
            const mfem::Array<int> & test_native_to_lexicographic = dynamic_cast<const mfem::TensorBasisElement *>(form.test_space_->GetBE(0))->GetDofMap();
            ::detail::apply_permutation(test_dofs, test_native_to_lexicographic);
          }

          if constexpr (trial::family != Family::HCURL) {
            const mfem::Array<int> & trial_native_to_lexicographic = dynamic_cast<const mfem::TensorBasisElement *>(form.trial_space_->GetBE(0))->GetDofMap();
            ::detail::apply_permutation(trial_dofs, trial_native_to_lexicographic);
          }

          for (int i = 0; i < dofs_per_test_boundary_element; i++) {
            for (int j = 0; j < dofs_per_trial_boundary_element; j++) {
              for (int k = 0; k < test_vdim; k++) {
                int test_vdof = form.test_space_->DofToVDof(::detail::get_index(test_dofs[i]), k);
                for (int l = 0; l < trial_vdim; l++) {
                  int trial_vdof = form.trial_space_->DofToVDof(::detail::get_index(trial_dofs[j]), l);
                  infos.push_back({test_vdof, trial_vdof, i + dofs_per_test_boundary_element * k, j + dofs_per_trial_boundary_element * l, b, ::detail::get_sign(test_dofs[i]) * ::detail::get_sign(trial_dofs[j]), on_boundary});
                }
              }
            }
          }
        }
      }

      std::sort(infos.begin(), infos.end());

      int nrows = form.test_space_->GetNDofs() * form.test_space_->GetVDim();
      row_ptr.resize(nrows + 1);
      std::vector < ::detail::signed_index > nonzero_ids(infos.size());

      int nnz = 0;
      row_ptr[0] = 0;
      col_ind.push_back(infos[0].global_col);
      nonzero_ids[0] = {0, infos[0].sign};

      for (size_t i = 1; i < infos.size(); i++) {
        // increment the nonzero count every time we find a new (i,j) pair
        nnz += (infos[i-1] != infos[i]);

        nonzero_ids[i] = {nnz, infos[i].sign};

        if (infos[i-1] != infos[i]) {
          col_ind.push_back(infos[i].global_col);
        }

        for (int j = infos[i-1].global_row; j < infos[i].global_row; j++) {
          row_ptr[j+1] = nonzero_ids[i];
        }
      }

      ++nnz;
      for (int j = infos.back().global_row; j < nrows; j++) {
        row_ptr[j+1] = nnz;
      }

      element_nonzero_LUT = Array3D<::detail::signed_index>(num_elements, dofs_per_test_element * test_vdim, dofs_per_trial_element * trial_vdim);
      boundary_element_nonzero_LUT = Array3D<::detail::signed_index>(num_boundary_elements, dofs_per_test_boundary_element * test_vdim, dofs_per_trial_boundary_element * trial_vdim);

      for (size_t i = 0; i < infos.size(); i++) {

        auto [_1, _2, local_row, local_col, element_id, _3, on_boundary] = infos[i];
        if (on_boundary) {
          boundary_element_nonzero_LUT(element_id, local_row, local_col) = nonzero_ids[i];
        } else {
          element_nonzero_LUT(element_id, local_row, local_col) = nonzero_ids[i];
        }
      }

      sparsity_pattern_initialized = true;

    }

    operator mfem::SparseMatrix() {

      if (!sparsity_pattern_initialized) initialize_sparsity_pattern();

      // the CSR graph (sparsity pattern) is reusable, so we cache
      // that and ask mfem to not free that memory in ~SparseMatrix()
      constexpr bool sparse_matrix_frees_graph_ptrs = false;

      // the CSR values are NOT reusable, so we pass ownership of
      // them to the mfem::SparseMatrix, to be freed in ~SparseMatrix()
      constexpr bool sparse_matrix_frees_values_ptr = true;

      constexpr bool col_ind_is_sorted = true;


      int nnz = row_ptr.back();
      double * values = new double[nnz]{};

      if (form.domain_integrals_.size() > 0) {
        int num_elements = form.test_space_->GetNE();
        int test_vdim  = form.test_space_->GetVDim();
        int trial_vdim = form.trial_space_->GetVDim();
        int dofs_per_test_element = form.test_space_->GetFE(0)->GetDof();
        int dofs_per_trial_element = form.trial_space_->GetFE(0)->GetDof();

        mfem::Vector element_matrices = form.ComputeElementMatrices();
        auto K_elem = mfem::Reshape(element_matrices.HostReadWrite(), dofs_per_test_element * test_vdim, dofs_per_trial_element * trial_vdim, num_elements);
        for (int e = 0; e < num_elements; e++) {
          for (int i = 0; i < dofs_per_test_element * test_vdim; i++) {
            for (int j = 0; j < dofs_per_trial_element * trial_vdim; j++) {
              auto [index, sign] = element_nonzero_LUT(e,i,j);
              values[index] += sign * K_elem(i,j,e);
            }
          }
        }
      }

      if (form.boundary_integrals_.size() > 0) {
        int num_boundary_elements = form.test_space_->GetNBE();
        int test_vdim  = form.test_space_->GetVDim();
        int trial_vdim = form.trial_space_->GetVDim();
        int dofs_per_test_boundary_element = form.test_space_->GetBE(0)->GetDof();
        int dofs_per_trial_boundary_element = form.trial_space_->GetBE(0)->GetDof();
 
        mfem::Vector boundary_element_matrices = form.ComputeBoundaryElementMatrices();
        auto K_elem = mfem::Reshape(boundary_element_matrices.HostReadWrite(), dofs_per_test_boundary_element * test_vdim, dofs_per_trial_boundary_element * trial_vdim, num_boundary_elements);
        for (int e = 0; e < num_boundary_elements; e++) {
          for (int i = 0; i < dofs_per_test_boundary_element * test_vdim; i++) {
            for (int j = 0; j < dofs_per_trial_boundary_element * trial_vdim; j++) {
              auto [index, sign] = boundary_element_nonzero_LUT(e,i,j);
              values[index] += sign * K_elem(i,j,e);
            }
          }
        }
      }

      return mfem::SparseMatrix(row_ptr.data(), col_ind.data(), values, Height(), Width(), sparse_matrix_frees_graph_ptrs, sparse_matrix_frees_values_ptr, col_ind_is_sorted);
      
    }

  private:

    /**
     * @brief The "parent" @p Functional to calculate gradients with
     */
    Functional<test(trial), execution_policy>& form;

    std::vector< int > row_ptr;
    std::vector< int > col_ind;

    Array3D< ::detail::signed_index > element_nonzero_LUT;
    Array3D< ::detail::signed_index > boundary_element_nonzero_LUT;

    mfem::SparseMatrix A;

    bool sparsity_pattern_initialized;
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

    if (boundary_integrals_.size() > 0) {
      G_trial_boundary_->Mult(input_L_, input_E_boundary_);

      output_E_boundary_ = 0.0;
      for (auto& integral : boundary_integrals_) {
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
  std::vector<DomainIntegral<test(trial), execution_policy> > domain_integrals_;

  /**
   * @brief The set of boundary integral (spatial_dim > geometric_dim)
   */
  std::vector<BoundaryIntegral<test(trial)> > boundary_integrals_;

  // simplex elements are currently not supported;
  static constexpr mfem::Element::Type supported_types[4] = {mfem::Element::POINT, mfem::Element::SEGMENT,
                                                             mfem::Element::QUADRILATERAL, mfem::Element::HEXAHEDRON};

  /**
   * @brief The gradient object used to implement @p GetGradient
   */
  mutable Gradient grad_;

  /**
   * @brief storage buffer for element stiffness matrices, used in ComputeElementMatrices() and
   * UpdateAssembledSparseMatrix()
   */
  mutable mfem::Vector K_e_;

  /**
   * @brief storage buffer for boundary element stiffness matrices, used in ComputeBoundaryElementMatrices() and
   * UpdateAssembledSparseMatrix()
   */
  mutable mfem::Vector K_b_;

  /**
   * @brief Local internal AssembledSparseMatrix storage for ComputeElementMatrices
   *
   * If unique_ptr is empty, construct AssembledSparseMatrix.
   *
   */
  std::unique_ptr<serac::mfem_ext::AssembledSparseMatrix> assembled_spmat_;


  template < typename T >
  friend typename Functional<T>::Gradient & grad(Functional<T> &);
};

template < typename T >
typename Functional<T>::Gradient & grad(Functional<T> & f) { return f.grad_; }

}  // namespace serac

#include "functional_qoi.inl"
