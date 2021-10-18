// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file functional_qoi.inl
 *
 * @brief a specialization of serac::Functional for quantities of interest
 */

namespace serac {

struct QoIProlongation : public mfem::Operator {
  QoIProlongation(MPI_Comm c) : mfem::Operator(1, 1), comm(c) {}

  void Mult(const mfem::Vector&, mfem::Vector&) const override
  {
    SLIC_ERROR_ROOT("QoIProlongation::Mult() is not defined");
  }

  void MultTranspose(const mfem::Vector& input, mfem::Vector& output) const override
  {
    MPI_Allreduce(&input[0], &output[0], 1, MPI_DOUBLE, MPI_SUM, comm);
  }

  MPI_Comm comm;
};

struct QoIElementRestriction : public mfem::Operator {
  QoIElementRestriction(int num_elements) : mfem::Operator(num_elements, 1) {}

  void Mult(const mfem::Vector&, mfem::Vector&) const override
  {
    SLIC_ERROR_ROOT("QoIElementRestriction::Mult() is not defined, exiting...");
  }

  void MultTranspose(const mfem::Vector& input, mfem::Vector& output) const override { output[0] = input.Sum(); }
};

/**
 * @brief a partial template specialization of Functional with test == double, implying "quantity of interest"
 */
template <typename trial, ExecutionSpace exec>
class Functional<double(trial), exec> : public mfem::Operator {
  using test = QOI;

public:
  /**
   * @brief Constructs using a @p mfem::ParFiniteElementSpace object corresponding to the trial space
   * @param[in] trial_fes The trial space
   */
  Functional(mfem::ParFiniteElementSpace* trial_fes)
      : Operator(1 /* the output of a QoI is a scalar */, trial_fes->GetTrueVSize()),
        trial_space_(trial_fes),
        P_test_(new QoIProlongation(trial_fes->GetParMesh()->GetComm())),
        G_test_(new QoIElementRestriction(trial_fes->GetParMesh()->GetNE())),
        P_trial_(trial_space_->GetProlongationMatrix()),
        G_trial_(trial_space_->GetElementRestriction(mfem::ElementDofOrdering::LEXICOGRAPHIC)),
        grad_(*this)
  {
    SLIC_ERROR_IF(!G_test_, "Couldn't retrieve element restriction operator for test space");
    SLIC_ERROR_IF(!G_trial_, "Couldn't retrieve element restriction operator for trial space");

    G_test_boundary_ = new QoIElementRestriction(trial_fes->GetParMesh()->GetNBE());

    // Ensure the mesh has the appropriate neighbor information before constructing the face restriction operators
    if (trial_space_) trial_space_->ExchangeFaceNbrData();

    // for now, limitations in mfem prevent us from implementing surface integrals for Hcurl test/trial space
    if (trial::family != Family::HCURL) {
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
      auto num_elements           = static_cast<size_t>(trial_space_->GetNE());
      auto ndof_per_test_element  = static_cast<size_t>(1);
      auto ndof_per_trial_element = static_cast<size_t>(trial_space_->GetFE(0)->GetDof() * trial_space_->GetVDim());
      element_gradients_          = Array<double, 3, exec>(num_elements, ndof_per_test_element, ndof_per_trial_element);
    }

    {
      auto num_bdr_elements           = static_cast<size_t>(trial_space_->GetNFbyType(mfem::FaceType::Boundary));
      auto ndof_per_test_bdr_element  = static_cast<size_t>(1);
      auto ndof_per_trial_bdr_element = static_cast<size_t>(trial_space_->GetBE(0)->GetDof() * trial_space_->GetVDim());
      bdr_element_gradients_ =
          Array<double, 3, exec>(num_bdr_elements, ndof_per_test_bdr_element, ndof_per_trial_bdr_element);
    }
  }

  ~Functional()
  {
    delete P_test_;
    delete G_test_;
    delete G_test_boundary_;
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

    SLIC_ERROR_ROOT_IF(dim != domain.Dimension(), "invalid mesh dimension for domain integral");
    for (int e = 0; e < num_elements; e++) {
      SLIC_ERROR_ROOT_IF(domain.GetElementType(e) != supported_types[dim], "Mesh contains unsupported element type");
    }

    const mfem::FiniteElement&   el = *trial_space_->GetFE(0);
    const mfem::IntegrationRule& ir = mfem::IntRules.Get(el.GetGeomType(), el.GetOrder() * 2);

    constexpr auto flags = mfem::GeometricFactors::COORDINATES | mfem::GeometricFactors::JACOBIANS;
    auto           geom  = domain.GetGeometricFactors(ir, flags);
    domain_integrals_.emplace_back(num_elements, geom->J, geom->X, Dimension<dim>{}, integrand);
  }

  /**
   * @brief Adds a boundary integral term to the weak formulation of the PDE
   * @tparam dim The dimension of the boundary element (1 for line, 2 for quad, etc)
   * @tparam lambda the type of the integrand functor: must implement operator() with an appropriate function signature
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
    auto num_bdr_elements = domain.GetNBE();
    if (num_bdr_elements == 0) return;

    SLIC_ERROR_ROOT_IF((dim + 1) != domain.Dimension(), "invalid mesh dimension for boundary integral");
    for (int e = 0; e < num_bdr_elements; e++) {
      SLIC_ERROR_ROOT_IF(domain.GetBdrElementType(e) != supported_types[dim], "Mesh contains unsupported element type");
    }

    const mfem::FiniteElement&   el = *trial_space_->GetBE(0);
    const mfem::IntegrationRule& ir = mfem::IntRules.Get(supported_types[dim], el.GetOrder() * 2);
    constexpr auto flags = mfem::FaceGeometricFactors::COORDINATES | mfem::FaceGeometricFactors::DETERMINANTS |
                           mfem::FaceGeometricFactors::NORMALS;

    // despite what their documentation says, mfem doesn't actually support the JACOBIANS flag.
    // this is currently a dealbreaker, as we need this information to do any calculations
    auto geom = domain.GetFaceGeometricFactors(ir, flags, mfem::FaceType::Boundary);
    bdr_integrals_.emplace_back(num_bdr_elements, geom->detJ, geom->X, geom->normal, Dimension<dim>{}, integrand);
  }

  /**
   * @brief Adds an area integral, i.e., over 2D elements in R^2
   * @tparam lambda the type of the integrand functor: must implement operator() with an appropriate function signature
   * @tparam qpt_data_type The type of the data to store for each quadrature point
   * @param[in] integrand The quadrature function
   * @param[in] domain The mesh to evaluate the integral on
   * @param[in] data The data structure containing per-quadrature-point data
   */
  template <typename lambda, typename qpt_data_type = void>
  void AddAreaIntegral(lambda&& integrand, mfem::Mesh& domain)
  {
    AddDomainIntegral(Dimension<2>{}, integrand, domain);
  }

  /**
   * @brief Adds a volume integral, i.e., over 3D elements in R^3
   * @tparam lambda the type of the integrand functor: must implement operator() with an appropriate function signature
   * @tparam qpt_data_type The type of the data to store for each quadrature point
   * @param[in] integrand The quadrature function
   * @param[in] domain The mesh to evaluate the integral on
   * @param[in] data The data structure containing per-quadrature-point data
   */
  template <typename lambda, typename qpt_data_type = void>
  void AddVolumeIntegral(lambda&& integrand, mfem::Mesh& domain)
  {
    AddDomainIntegral(Dimension<3>{}, integrand, domain);
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
  void Mult(const mfem::Vector& input_T, mfem::Vector& output_T) const
  {
    Evaluation<Operation::Mult>(input_T, output_T);
  }

  /**
   * @brief Alias for @p Mult that uses a return value instead of an output parameter
   * @param[in] input_T The input vector
   */
  double operator()(const mfem::Vector& input_T) const
  {
    Evaluation<Operation::Mult>(input_T, my_output_T_);
    return my_output_T_[0];
  }

  /**
   * @brief Obtains the gradients for all the constituent integrals
   * @param[in] input_T The input vector
   * @param[out] output_T The output vector
   * @see DomainIntegral::GradientMult, BoundaryIntegral::GradientMult
   */
  void GradientMult(const mfem::Vector& input_T, mfem::Vector& output_T) const
  {
    Evaluation<Operation::GradientMult>(input_T, output_T);
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
   * @brief mfem::Operator that produces the gradient of a @p Functional from a @p Mult
   */
  class Gradient {
  public:
    /**
     * @brief Constructs a Gradient wrapper that references a parent @p Functional
     * @param[in] f The @p Functional to use for gradient calculations
     */
    Gradient(Functional<double(trial)>& f)
        : form_(f),
          lookup_tables(*(f.trial_space_)),
          gradient_L_(f.trial_space_->GetVSize()),
          gradient_T_(f.trial_space_->NewTrueDofVector())
    {
    }

    void Mult(const mfem::Vector& x, mfem::Vector& y) const { form_.GradientMult(x, y); }

    double operator()(const mfem::Vector& x) const
    {
      mfem::Vector y(1);
      form_.GradientMult(x, y);
      return y[0];
    }

    operator mfem::HypreParVector &()
    {
      gradient_L_ = 0.0;

      if (form_.domain_integrals_.size() > 0) {
        auto& K_elem = form_.element_gradients_;
        auto& LUT    = lookup_tables.element_dofs_;

        zero_out(K_elem);
        for (auto& domain : form_.domain_integrals_) {
          domain.ComputeElementGradients(view(K_elem));
        }

        for (size_t e = 0; e < K_elem.size(0); e++) {
          for (size_t j = 0; j < K_elem.size(2); j++) {
            auto [index, sign] = LUT(e, j);
            gradient_L_[index] += sign * K_elem(e, 0, j);
          }
        }
      }

      if (form_.bdr_integrals_.size() > 0) {
        auto& K_belem = form_.bdr_element_gradients_;
        auto& LUT     = lookup_tables.bdr_element_dofs_;

        zero_out(K_belem);
        for (auto& boundary : form_.bdr_integrals_) {
          boundary.ComputeElementGradients(view(K_belem));
        }

        for (size_t e = 0; e < K_belem.size(0); e++) {
          for (size_t j = 0; j < K_belem.size(2); j++) {
            auto [index, sign] = LUT(e, j);
            gradient_L_[index] += sign * K_belem(e, 0, j);
          }
        }
      }

      form_.P_trial_->MultTranspose(gradient_L_, *gradient_T_);

      return *gradient_T_;
    }

  private:
    /**
     * @brief The "parent" @p Functional to calculate gradients with
     */
    Functional<double(trial), exec>& form_;

    DofNumbering lookup_tables;

    mfem::Vector gradient_L_;

    std::unique_ptr<mfem::HypreParVector> gradient_T_;
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
   * @brief The input set of per-boundary-element DOF values
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

  /// @brief The gradient object used to implement @p GetGradient
  mutable Gradient grad_;

  /// @brief array that stores each element's gradient of the residual w.r.t. trial values
  Array<double, 3, exec> element_gradients_;

  /// @brief array that stores each boundary element's gradient of the residual w.r.t. trial values
  Array<double, 3, exec> bdr_element_gradients_;

  template <typename T>
  friend typename Functional<T>::Gradient& grad(Functional<T>&);
};

}  // namespace serac
