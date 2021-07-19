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
#include "serac/physics/utilities/functional/integral.hpp"
#include "serac/numerics/assembled_sparse_matrix.hpp"

namespace serac {

struct QoIProlongation : public mfem::Operator { 
  QoIProlongation(MPI_Comm c) : mfem::Operator(1, 1), comm(c) {}

  void Mult(const mfem::Vector&, mfem::Vector&) const override {
    std::cout << "QoIProlongation::Mult() is not defined, exiting..." << std::endl;
    std::exit(1);
  }

  void MultTranspose(const mfem::Vector& input, mfem::Vector& output) const override {
    MPI_Allreduce (&input[0], &output[0], 1, MPI_DOUBLE, MPI_SUM, comm);
  }

  MPI_Comm comm;
};

struct QoIElementRestriction : public mfem::Operator { 
  QoIElementRestriction(int num_elements) : mfem::Operator(num_elements, 1) {}

  void Mult(const mfem::Vector&, mfem::Vector&) const override {
    std::cout << "QoIElementRestriction::Mult() is not defined, exiting..." << std::endl;
    std::exit(1);
  }

  void MultTranspose(const mfem::Vector& input, mfem::Vector& output) const override {
    output[0] = input.Sum();
  }
};

/// @cond
template <typename T>
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
template <typename test, typename trial>
class Functional<test(trial)> : public mfem::Operator {

  using return_type = std::conditional_t< is_qoi_v<test>, double, mfem::Vector & >;

 private:
  void InitializeVectors() {
    SLIC_ERROR_IF(!G_test_, "Couldn't retrieve element restriction operator for test space");
    SLIC_ERROR_IF(!G_trial_, "Couldn't retrieve element restriction operator for trial space");
    input_L_.SetSize(P_trial_->Height(), mfem::Device::GetMemoryType());
    input_E_.SetSize(G_trial_->Height(), mfem::Device::GetMemoryType());
    output_E_.SetSize(G_test_->Height(), mfem::Device::GetMemoryType());
    output_L_.SetSize(P_test_->Height(), mfem::Device::GetMemoryType());
    my_output_T_.SetSize(Height(), mfem::Device::GetMemoryType());
    dummy_.SetSize(Width(), mfem::Device::GetMemoryType());
  }

 public:

  /**
   * @brief Constructs using a @p mfem::ParFiniteElementSpace object corresponding to the trial space
   * @param[in] trial_fes The trial space
   */
  Functional(mfem::ParFiniteElementSpace* trial_fes)
      : Operator(1 /* the output of a QoI is a scalar */, trial_fes->GetTrueVSize()),
        test_space_(nullptr),
        trial_space_(trial_fes),
        P_test_(new QoIProlongation(trial_fes->GetParMesh()->GetComm())),
        G_test_(new QoIElementRestriction(trial_fes->GetParMesh()->GetNE())),
        P_trial_(trial_space_->GetProlongationMatrix()),
        G_trial_(trial_space_->GetElementRestriction(mfem::ElementDofOrdering::LEXICOGRAPHIC)),
        grad_(*this) {
    static_assert(is_qoi_v<test>, "Functional::Functional(ParFES) constructor only enabled for QoI test space");
    InitializeVectors();
  }

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
        grad_(*this) {
    static_assert(!is_qoi_v<test>, "Functional::Functional(ParFES, ParFES) constructor only enabled for non-QoI test spaces");
    InitializeVectors();
  }

  /**
   * @brief Adds an integral term to the weak formulation of the PDE
   * @tparam geometry_dim The dimension of the element (2 for quad, 3 for hex, etc)
   * @tparam spatial_dim The full dimension of the mesh
   * @tparam lambda the type of the integrand functor: must implement operator() with an appropriate function signature
   * @param[in] integrand The user-provided quadrature function, see @p Integral
   * @param[in] domain The domain on which to evaluate the integral
   * @note The @p Dimension parameters are used to assist in the deduction of the @a geometry_dim
   * and @a spatial_dim template parameter
   */
  template <int geometry_dim, int spatial_dim, typename lambda>
  void AddIntegral(Dimension<geometry_dim>, Dimension<spatial_dim>, lambda&& integrand, mfem::Mesh& domain)
  {
    if constexpr (geometry_dim == spatial_dim) {
      auto num_elements = domain.GetNE();
      SLIC_ERROR_IF(num_elements == 0, "Mesh has no elements");

      auto dim = domain.Dimension();
      for (int e = 0; e < num_elements; e++) {
        SLIC_ERROR_IF(domain.GetElementType(e) != supported_types[dim], "Mesh contains unsupported element type");
      }

      const mfem::FiniteElement&   el = *trial_space_->GetFE(0);
      const mfem::IntegrationRule& ir = mfem::IntRules.Get(el.GetGeomType(), el.GetOrder() * 2);

      constexpr auto flags = mfem::GeometricFactors::COORDINATES | mfem::GeometricFactors::JACOBIANS;
      auto           geom  = domain.GetGeometricFactors(ir, flags);
      domain_integrals_.emplace_back(num_elements, geom->J, geom->X, Dimension<geometry_dim>{},
                                     Dimension<spatial_dim>{}, integrand);
      return;
    }
#ifdef ENABLE_BOUNDARY_INTEGRALS
    else if constexpr ((geometry_dim + 1) == spatial_dim) {
      // TODO: fix mfem::FaceGeometricFactors
      constexpr bool always_false = (geometry_dim == spatial_dim);
      static_assert(always_false,
                    "unsupported integral dimensionality due to unimplemented features in mfem::FaceGeometricFactors");

      auto num_boundary_elements = domain.GetNBE();
      SLIC_ERROR_IF(num_boundary_elements == 0, "Mesh has no boundary elements");

      auto dim = domain.Dimension();
      SLIC_ERROR_IF(dim != spatial_dim, "Error: invalid mesh dimension for integral");
      for (int e = 0; e < num_boundary_elements; e++) {
        SLIC_ERROR_IF(domain.GetBdrElementType(e) != supported_types[dim - 1],
                      "Mesh contains unsupported element type");
      }

      const mfem::FiniteElement&   el = *test_space_->GetBE(0);
      const mfem::IntegrationRule& ir = mfem::IntRules.Get(el.GetGeomType(), el.GetOrder() * 2);
      constexpr auto flags            = mfem::FaceGeometricFactors::COORDINATES | mfem::FaceGeometricFactors::JACOBIANS;

      // despite what their documentation says, mfem doesn't actually support the JACOBIANS flag.
      // this is currently a dealbreaker, as we need this information to do any calculations
      auto geom = domain.GetFaceGeometricFactors(ir, flags, mfem::FaceType::Boundary);
      boundary_integrals_.emplace_back(num_boundary_elements, geom->J, geom->X, Dimension<geometry_dim>{},
                                       Dimension<spatial_dim>{}, integrand);
      return;
    }
#endif

    else {
      // if static_assert has a literal 'false' for its first arg,
      // it will trigger even when this branch isn't selected. So,
      // we define an expression that is always false (see first
      // if constexpr expression) to work around this limitation
      constexpr bool always_false = (geometry_dim == spatial_dim);
      static_assert(always_false, "unsupported integral dimensionality");
    }
  }

  /**
   * @brief Adds an area integral, i.e., over 2D elements in R^2 space
   * @tparam lambda the type of the integrand functor: must implement operator() with an appropriate function signature
   * @param[in] integrand The quadrature function
   * @param[in] domain The mesh to evaluate the integral on
   */
  template <typename lambda>
  void AddAreaIntegral(lambda&& integrand, mfem::Mesh& domain)
  {
    AddIntegral(Dimension<2>{} /* geometry */, Dimension<2>{} /* spatial */, integrand, domain);
  }

  /**
   * @brief Adds a volume integral, i.e., over 3D elements in R^3 space
   * @tparam lambda the type of the integrand functor: must implement operator() with an appropriate function signature
   * @param[in] integrand The quadrature function
   * @param[in] domain The mesh to evaluate the integral on
   */
  template <typename lambda>
  void AddVolumeIntegral(lambda&& integrand, mfem::Mesh& domain)
  {
    AddIntegral(Dimension<3>{} /* geometry */, Dimension<3>{} /* spatial */, integrand, domain);
  }

  /**
   * @brief Adds a domain integral
   * @tparam d The dimension of the elements *and* the space they're embedded in
   * @tparam lambda the type of the integrand functor: must implement operator() with an appropriate function signature
   * @param[in] integrand The quadrature function
   * @param[in] domain The mesh to evaluate the integral on
   */
  template <int d, typename lambda>
  void AddDomainIntegral(Dimension<d>, lambda&& integrand, mfem::Mesh& domain)
  {
    AddIntegral(Dimension<d>{} /* geometry */, Dimension<d>{} /* spatial */, integrand, domain);
  }

#ifdef ENABLE_BOUNDARY_INTEGRALS
  /**
   * @brief Adds a surface integral, i.e., over 2D elements in R^3 space
   * @tparam lambda the type of the integrand functor: must implement operator() with an appropriate function signature
   * @param[in] integrand The quadrature function
   * @param[in] domain The mesh to evaluate the integral on
   */
  template <typename lambda>
  void AddSurfaceIntegral(lambda&& integrand, mfem::Mesh& domain)
  {
    AddIntegral(Dimension<2>{} /* geometry */, Dimension<3>{} /* spatial */, integrand, domain);
  }
#endif

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
  return_type operator()(const mfem::Vector& input_T) const
  {
    Evaluation<Operation::Mult>(input_T, my_output_T_);
    if constexpr (is_qoi_v<test>) {
      return my_output_T_[0];
    } else {
      return my_output_T_;
    }
  }

  /**
   * @brief Obtains the gradients for all the constituent integrals
   * @param[in] input_T The input vector
   * @param[out] output_T The output vector
   * @see Integral::GradientMult
   */
  virtual void GradientMult(const mfem::Vector& input_T, mfem::Vector& output_T) const
  {
    Evaluation<Operation::GradientMult>(input_T, output_T);
  }

  /**
   * @brief Obtains the element stiffness matrix reshaped a mfem::Vector
   * @returns A mfem::Vector containing the assembled element stiffness matrix (test_dim * test_ndof, trial_dim
   * * trial_ndof, nelem)
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
    for (auto domain : domain_integrals_) domain.ComputeElementMatrices(K_e_);

    return K_e_;
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
  public:
    /**
     * @brief Constructs a Gradient wrapper that references a parent @p Functional
     * @param[in] f The @p Functional to use for gradient calculations
     */
    Gradient(Functional& f) : mfem::Operator(f.Height()), form(f){};

    virtual void Mult(const mfem::Vector& x, mfem::Vector& y) const override { form.GradientMult(x, y); }

  private:
    /**
     * @brief The "parent" @p Functional to calculate gradients with
     */
    Functional<test(trial)>& form;
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

      output_E_.Print(std::cout);

      // scatter-add to compute residuals on the local processor
      G_test_->MultTranspose(output_E_, output_L_);
    }

#ifdef ENABLE_BOUNDARY_INTEGRALS
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

      // scatter-add to compute residuals on the local processor
      G_test_boundary_->MultTranspose(output_E_boundary_, output_L_boundary_);

      output_L_ += output_L_boundary_;
    }
#endif

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

#ifdef ENABLE_BOUNDARY_INTEGRALS
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
  mutable mfem::Vector output_L_boundary_
#endif

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

#ifdef ENABLE_BOUNDARY_INTEGRALS
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
#endif

  /**
   * @brief The set of domain integrals (spatial_dim == geometric_dim)
   */
  std::vector<Integral<test(trial)> > domain_integrals_;

#ifdef ENABLE_BOUNDARY_INTEGRALS
  /**
   * @brief The set of boundary integral (spatial_dim > geometric_dim)
   */
  std::vector<Integral<test(trial)> > boundary_integrals_;
#endif

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
   * @brief Local internal AssembledSparseMatrix storage for ComputeElementMatrices
   *
   * If unique_ptr is empty, construct AssembledSparseMatrix.
   *
   */
  std::unique_ptr<serac::mfem_ext::AssembledSparseMatrix> assembled_spmat_;
};

}  // namespace serac
