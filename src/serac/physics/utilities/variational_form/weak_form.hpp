#pragma once

#include "mfem.hpp"
#include "mfem/general/forall.hpp"

#include "serac/physics/utilities/variational_form/tensor.hpp"
#include "serac/physics/utilities/variational_form/quadrature.hpp"
#include "serac/physics/utilities/variational_form/finite_element.hpp"
#include "serac/physics/utilities/variational_form/tuple_arithmetic.hpp"
#include "serac/physics/utilities/variational_form/integral.hpp"

namespace serac {

/// @cond
template <typename T>
class WeakForm;
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
 * WeakForm< H1(Hcurl) > my_residual;
 * @endcode
 *
 * @tparam test The space of test functions to use
 * @tparam trial The space of trial functions to use
 *
 * To use this class, you use the methods @p WeakForm::Add****Integral(integrand,domain_of_integration)
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
class WeakForm<test(trial)> : public mfem::Operator {
public:
  /**
   * @brief Constructs using @p mfem::ParFiniteElementSpace objects corresponding to the test/trial spaces
   * @param[in] test_fes The test space
   * @param[in] trial_fes The trial space
   */
  WeakForm(mfem::ParFiniteElementSpace* test_fes, mfem::ParFiniteElementSpace* trial_fes)
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

    input_L_.SetSize(P_test_->Height(), mfem::Device::GetMemoryType());
    input_E_.SetSize(G_test_->Height(), mfem::Device::GetMemoryType());

    output_E_.SetSize(G_trial_->Height(), mfem::Device::GetMemoryType());
    output_L_.SetSize(P_trial_->Height(), mfem::Device::GetMemoryType());

    my_output_T.SetSize(test_fes->GetTrueVSize(), mfem::Device::GetMemoryType());

    dummy_.SetSize(trial_fes->GetTrueVSize(), mfem::Device::GetMemoryType());
  }

  /**
   * @brief Adds an integral term to the weak formulation of the PDE
   * @tparam geometry_dim The dimension of the element (2 for quad, 3 for hex, etc)
   * @tparam spatial_dim The full dimension of the mesh
   * @param[in] integrand The user-provided quadrature function, see @p Integral
   * @param[in] domain The domain on which to evaluate the integral
   * @note The @p Dimension parameters are used to assist in the deduction of the @a geometry_dim
   * and @a spatial_dim template parameter
   */
  template <int geometry_dim, int spatial_dim, typename lambda>
  void AddIntegral(Dimension<geometry_dim>, Dimension<spatial_dim>, lambda&& integrand, mfem::Mesh& domain)
  {
    auto num_elements = domain.GetNE();
    SLIC_ERROR_IF(num_elements == 0, "Mesh has no elements");

    auto dim = domain.Dimension();
    for (int e = 0; e < num_elements; e++) {
      SLIC_ERROR_IF(domain.GetElementType(e) != supported_types[dim], "Mesh contains unsupported element type");
    }

    if constexpr (geometry_dim == spatial_dim) {
      const mfem::FiniteElement&   el = *test_space_->GetFE(0);
      const mfem::IntegrationRule& ir = mfem::IntRules.Get(el.GetGeomType(), el.GetOrder() * 2);
      auto                         geom =
          domain.GetGeometricFactors(ir, mfem::GeometricFactors::COORDINATES | mfem::GeometricFactors::JACOBIANS);
      domain_integrals_.emplace_back(num_elements, geom->J, geom->X, Dimension<geometry_dim>{},
                                     Dimension<spatial_dim>{}, integrand);
      return;
    } else if constexpr ((geometry_dim + 1) == spatial_dim) {
      const mfem::FiniteElement&   el = *test_space_->GetBE(0);
      const mfem::IntegrationRule& ir = mfem::IntRules.Get(el.GetGeomType(), el.GetOrder() * 2);
      constexpr auto flags = mfem::FaceGeometricFactors::COORDINATES | mfem::FaceGeometricFactors::JACOBIANS |
                             mfem::FaceGeometricFactors::NORMALS;
      auto geom = domain.GetFaceGeometricFactors(ir, flags, mfem::FaceType::Boundary);
      boundary_integrals_.emplace_back(num_elements, geom->J, geom->X, Dimension<geometry_dim>{},
                                       Dimension<spatial_dim>{}, integrand);
      return;
    } else if constexpr (true) {
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
   * @param[in] integrand The quadrature function
   * @param[in] domain The mesh to evaluate the integral on
   */
  template <int d, typename lambda>
  void AddDomainIntegral(Dimension<d>, lambda&& integrand, mfem::Mesh& domain)
  {
    AddIntegral(Dimension<d>{} /* geometry */, Dimension<d>{} /* spatial */, integrand, domain);
  }

  /**
   * @brief Adds a surface integral, i.e., over 2D elements in R^3 space
   * @param[in] integrand The quadrature function
   * @param[in] domain The mesh to evaluate the integral on
   */
  template <typename lambda>
  void AddSurfaceIntegral(lambda&& integrand, mfem::Mesh& domain)
  {
    AddIntegral(Dimension<2>{} /* geometry */, Dimension<3>{} /* spatial */, integrand, domain);
  }

  void Mult(const mfem::Vector& input_T, mfem::Vector& output_T) const override
  {
    Evaluation<Operation::Mult>(input_T, output_T);
  }

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
    Evaluation<Operation::Mult>(input_T, my_output_T);
    return my_output_T;
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
   * @brief Applies an essential boundary condition to the attributes specified by @a ess_attr
   * @param[in] ess_attr The mesh attributes to apply the BC to
   *
   * @note This gets more interesting when having more than one trial space
   */
  void SetEssentialBC(const mfem::Array<int>& ess_attr)
  {
    static_assert(std::is_same_v<test, trial>, "can't specify essential bc on incompatible spaces");
    test_space_->GetEssentialTrueDofs(ess_attr, ess_tdof_list_);
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
   * @brief Lightweight shim for mfem::Operator that produces the gradient of a @p WeakForm from a @p Mult
   */
  class Gradient : public mfem::Operator {
  public:
    /**
     * @brief Constructs a Gradient wrapper that references a parent @p WeakForm
     * @param[in] f The @p WeakForm to use for gradient calculations
     */
    Gradient(WeakForm& f) : mfem::Operator(f.Height()), form(f){};

    virtual void Mult(const mfem::Vector& x, mfem::Vector& y) const override { form.GradientMult(x, y); }

  private:
    /**
     * @brief The "parent" @p WeakForm to calculate gradients with
     */
    WeakForm<test(trial)>& form;
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

    // get the values for each element on the local processor
    G_trial_->Mult(input_L_, input_E_);

    // compute residual contributions at the element level and sum them
    //
    // note: why should we serialize these integral evaluations?
    //       these could be performed in parallel and merged in the reduction process
    //
    // TODO investigate performance of alternative implementation described above
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
  // ^ FIXME: Do we need separate input/output vectors for L?
  /**
   * @brief The input set of per-element DOF values
   */
  mutable mfem::Vector input_E_;
  /**
   * @brief The output set of per-element DOF values
   */
  mutable mfem::Vector output_E_;
  /**
   * @brief The set of true DOF values, used as a scratchpad for @p operator()
   */
  mutable mfem::Vector my_output_T;
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
   * @brief The set of domain integrals (spatial_dim == geometric_dim)
   */
  std::vector<Integral<test(trial)> > domain_integrals_;
  /**
   * @brief The set of boundary integral (spatial_dim > geometric_dim)
   */
  std::vector<Integral<test(trial)> > boundary_integrals_;

  // simplex elements are currently not supported;
  static constexpr mfem::Element::Type supported_types[4] = {mfem::Element::POINT, mfem::Element::SEGMENT,
                                                             mfem::Element::QUADRILATERAL, mfem::Element::HEXAHEDRON};

  /**
   * @brief The gradient object used to implement @p GetGradient
   */
  mutable Gradient grad_;
};

}  // namespace serac
