// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
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

/**
 * @brief this class behaves like a Prolongation operator, except is specialized for
 * the case of a quantity of interest. The action of its MultTranspose() operator (the
 * only thing it is used for) sums the values from different processors.
 */
struct QoIProlongation : public mfem::Operator {
  /// @brief create a QoIProlongation for a Quantity of Interest
  QoIProlongation(MPI_Comm c) : mfem::Operator(1, 1), comm(c) {}

  /// @brief unimplemented: do not use
  void Mult(const mfem::Vector&, mfem::Vector&) const override
  {
    SLIC_ERROR_ROOT("QoIProlongation::Mult() is not defined");
  }

  /// @brief set the value of output to the distributed sum over input values from different processors
  void MultTranspose(const mfem::Vector& input, mfem::Vector& output) const override
  {
    MPI_Allreduce(&input[0], &output[0], 1, MPI_DOUBLE, MPI_SUM, comm);
  }

  MPI_Comm comm;  ///< MPI communicator used to carry out the distributed reduction
};

/**
 * @brief this class behaves like a Restriction operator, except is specialized for
 * the case of a quantity of interest. The action of its MultTranspose() operator (the
 * only thing it is used for) sums the values on this local processor.
 */
struct QoIElementRestriction : public mfem::Operator {
  /// @brief create a QoIElementRestriction for a Quantity of Interest
  QoIElementRestriction(int num_elements) : mfem::Operator(num_elements, 1) {}

  /// @brief unimplemented: do not use
  void Mult(const mfem::Vector&, mfem::Vector&) const override
  {
    SLIC_ERROR_ROOT("QoIElementRestriction::Mult() is not defined, exiting...");
  }

  /// @brief set the value of output to the sum of the values of input
  void MultTranspose(const mfem::Vector& input, mfem::Vector& output) const override { output[0] = input.Sum(); }
};

/**
 * @brief a partial template specialization of Functional with test == double, implying "quantity of interest"
 */
template <typename... trials, ExecutionSpace exec>
class Functional<double(trials...), exec> {
  using test = QOI;
  static constexpr tuple<trials...> trial_spaces{};
  static constexpr uint32_t         num_trial_spaces = sizeof...(trials);
  static constexpr auto             Q                = std::max({test::order, trials::order...}) + 1;

  static constexpr mfem::Geometry::Type elem_geom[4] = {mfem::Geometry::INVALID, mfem::Geometry::SEGMENT,
                                                        mfem::Geometry::SQUARE, mfem::Geometry::CUBE};

  class Gradient;

  // clang-format off
  template <int i> 
  struct operator_paren_return {
    using type = typename std::conditional<
        i >= 0,                           // if `i` is greater than or equal to zero,
        serac::tuple<double&, Gradient&>, // then we return the value and the derivative w.r.t arg `i`
        double                            // otherwise, we just return the value
        >::type;
  };
  // clang-format on

public:
  /**
   * @brief Constructs using a @p mfem::ParFiniteElementSpace object corresponding to the trial space
   * @param[in] trial_fes The trial space
   */
  Functional(std::array<const mfem::ParFiniteElementSpace*, num_trial_spaces> trial_fes) : trial_space_(trial_fes)
  {
    int dim = trial_fes[0]->GetMesh()->Dimension();

    for (uint32_t i = 0; i < num_trial_spaces; i++) {
      P_trial_[i] = trial_space_[i]->GetProlongationMatrix();

      input_L_[i].SetSize(P_trial_[i]->Height(), mfem::Device::GetMemoryType());

      G_trial_[i] = ElementRestriction(trial_fes[i], elem_geom[dim]);
      input_E_[i].SetSize(int(G_trial_[i].ESize()), mfem::Device::GetMemoryType());

      G_trial_boundary_[i] = ElementRestriction(trial_fes[i], elem_geom[dim - 1], FaceType::BOUNDARY);
      input_E_boundary_[i].SetSize(int(G_trial_boundary_[i].ESize()), mfem::Device::GetMemoryType());

      // create the gradient operators for each trial space
      grad_.emplace_back(*this, i);
    }

    P_test_          = new QoIProlongation(trial_fes[0]->GetParMesh()->GetComm());
    G_test_          = new QoIElementRestriction(trial_fes[0]->GetParMesh()->GetNE());
    G_test_boundary_ = new QoIElementRestriction(trial_fes[0]->GetParMesh()->GetNBE());

    output_E_.SetSize(G_test_->Height(), mfem::Device::GetMemoryType());
    output_E_boundary_.SetSize(G_test_boundary_->Height(), mfem::Device::GetMemoryType());

    output_L_.SetSize(P_test_->Height(), mfem::Device::GetMemoryType());
    output_L_boundary_.SetSize(P_test_->Height(), mfem::Device::GetMemoryType());

    output_T_.SetSize(1, mfem::Device::GetMemoryType());

    auto num_elements     = static_cast<size_t>(G_trial_[0].num_elements);
    auto num_bdr_elements = static_cast<size_t>(G_trial_boundary_[0].num_elements);
    for (uint32_t i = 0; i < num_trial_spaces; i++) {
      auto ndof_per_trial_element = static_cast<size_t>(G_trial_[i].nodes_per_elem * G_trial_[i].components);
      auto ndof_per_trial_bdr_element =
          static_cast<size_t>(G_trial_boundary_[i].nodes_per_elem * G_trial_boundary_[i].components);

      element_gradients_[i]     = ExecArray<double, 3, exec>(num_elements, 1, ndof_per_trial_element);
      bdr_element_gradients_[i] = ExecArray<double, 3, exec>(num_bdr_elements, 1, ndof_per_trial_bdr_element);
    }
  }

  /// @brief destructor: deallocate the mfem::Operators that we're responsible for
  ~Functional()
  {
    delete P_test_;
    delete G_test_;
    delete G_test_boundary_;
  }

  /**
   * @brief Adds a domain integral term to the Functional object
   * @tparam dim The dimension of the element (2 for quad, 3 for hex, etc)
   * @tparam lambda the type of the integrand functor: must implement operator() with an appropriate function signature
   * @tparam qpt_data_type The type of the data to store for each quadrature point
   * @param[in] integrand The user-provided quadrature function, see @p Integral
   * @param[in] domain The domain on which to evaluate the integral
   * @param[in] qdata The data structure containing per-quadrature-point data
   * @note The @p Dimension parameters are used to assist in the deduction of the @a geometry_dim
   * and @a spatial_dim template parameter
   */
  template <int dim, int... args, typename lambda, typename qpt_data_type = Nothing>
  void AddDomainIntegral(Dimension<dim>, DependsOn<args...>, lambda&& integrand, mfem::Mesh& domain,
                         std::shared_ptr<QuadratureData<qpt_data_type>> qdata = NoQData)
  {
    auto num_elements = domain.GetNE();
    if (num_elements == 0) return;

    SLIC_ERROR_ROOT_IF(dim != domain.Dimension(), "invalid mesh dimension for domain integral");
    for (int e = 0; e < num_elements; e++) {
      SLIC_ERROR_ROOT_IF(domain.GetElementType(e) != supported_types[dim], "Mesh contains unsupported element type");
    }

    // this is a temporary measure to check correctness for the replacement "GeometricFactor"
    // kernels, must be fixed before merging!
    auto* geom = new serac::GeometricFactors(&domain, Q, elem_geom[domain.Dimension()]);

    auto selected_trial_spaces = serac::make_tuple(serac::get<args>(trial_spaces)...);

    domain_integrals_.emplace_back(test{}, selected_trial_spaces, num_elements, geom->J, geom->X, Dimension<dim>{},
                                   integrand, qdata, std::vector<int>{args...});
  }

  /**
   * @tparam dim The dimension of the boundary element (1 for line, 2 for quad, etc)
   * @tparam lambda the type of the integrand functor: must implement operator() with an appropriate function signature
   * @param[in] integrand The user-provided quadrature function, see @p Integral
   * @param[in] domain The domain on which to evaluate the integral
   *
   * @brief Adds a boundary integral term to the Functional object
   *
   * @note The @p Dimension parameters are used to assist in the deduction of the @a geometry_dim
   * and @a spatial_dim template parameter
   */
  template <int dim, int... args, typename lambda, typename qpt_data_type = void>
  void AddBoundaryIntegral(Dimension<dim>, DependsOn<args...>, lambda&& integrand, mfem::Mesh& domain)
  {
    // TODO: fix mfem::FaceGeometricFactors
    auto num_bdr_elements = domain.GetNBE();
    if (num_bdr_elements == 0) return;

    SLIC_ERROR_ROOT_IF((dim + 1) != domain.Dimension(), "invalid mesh dimension for boundary integral");
    for (int e = 0; e < num_bdr_elements; e++) {
      SLIC_ERROR_ROOT_IF(domain.GetBdrElementType(e) != supported_types[dim], "Mesh contains unsupported element type");
    }

    // this is a temporary measure to check correctness for the replacement "GeometricFactor"
    // kernels, must be fixed before merging!
    auto geom = new serac::GeometricFactors(&domain, Q, elem_geom[dim], FaceType::BOUNDARY);

    auto selected_trial_spaces = serac::make_tuple(serac::get<args>(trial_spaces)...);

    bdr_integrals_.emplace_back(test{}, selected_trial_spaces, num_bdr_elements, geom->J, geom->X, Dimension<dim>{},
                                integrand, std::vector<int>{args...});
  }

  /**
   * @tparam lambda the type of the integrand functor: must implement operator() with an appropriate function signature
   * @tparam qpt_data_type The type of the data to store for each quadrature point
   * @param[in] which_args a tag type used to indicate which trial spaces are required by this calculation
   * @param[in] integrand The quadrature function
   * @param[in] domain The mesh to evaluate the integral on
   * @param[in] data The data structure containing per-quadrature-point data
   *
   * @brief Adds an area integral, i.e., over 2D elements in R^2
   */
  template <int... args, typename lambda, typename qpt_data_type = Nothing>
  void AddAreaIntegral(DependsOn<args...> which_args, lambda&& integrand, mfem::Mesh& domain,
                       std::shared_ptr<QuadratureData<qpt_data_type>>& data = NoQData)
  {
    AddDomainIntegral(Dimension<2>{}, which_args, integrand, domain, data);
  }

  /**
   * @tparam lambda the type of the integrand functor: must implement operator() with an appropriate function signature
   * @tparam qpt_data_type The type of the data to store for each quadrature point
   * @param[in] which_args a tag type used to indicate which trial spaces are required by this calculation
   * @param[in] integrand The quadrature function
   * @param[in] domain The mesh to evaluate the integral on
   * @param[in] data The data structure containing per-quadrature-point data
   *
   * @brief Adds a volume integral, i.e., over 3D elements in R^3
   */
  template <int... args, typename lambda, typename qpt_data_type = Nothing>
  void AddVolumeIntegral(DependsOn<args...> which_args, lambda&& integrand, mfem::Mesh& domain,
                         std::shared_ptr<QuadratureData<qpt_data_type>>& data = NoQData)
  {
    AddDomainIntegral(Dimension<3>{}, which_args, integrand, domain, data);
  }

  /// @brief alias for Functional::AddBoundaryIntegral(Dimension<2>{}, integrand, domain);
  template <int... args, typename lambda>
  void AddSurfaceIntegral(DependsOn<args...> which_args, lambda&& integrand, mfem::Mesh& domain)
  {
    AddBoundaryIntegral(Dimension<2>{}, which_args, integrand, domain);
  }

  /**
   * @brief this function computes the directional derivative of the quantity of interest functional
   *
   * @param input_T a T-vector to apply the action of gradient to
   * @param which describes which trial space input_T corresponds to
   *
   * note: it accepts exactly `num_trial_spaces` arguments of type mfem::Vector. Additionally, one of those
   * arguments may be a dual_vector, to indicate that Functional::operator() should not only evaluate the
   * element calculations, but also differentiate them w.r.t. the specified dual_vector argument
   */
  double ActionOfGradient(const mfem::Vector& input_T, std::size_t which) const
  {
    P_trial_[which]->Mult(input_T, input_L_[which]);

    output_L_ = 0.0;
    if (domain_integrals_.size() > 0) {
      // get the values for each element on the local processor
      G_trial_[which].Gather(input_L_[which], input_E_[which]);

      // compute residual contributions at the element level and sum them

      output_E_ = 0.0;
      for (auto& integral : domain_integrals_) {
        integral.GradientMult(input_E_[which], output_E_, which);
      }

      // scatter-add to compute residuals on the local processor
      G_test_->MultTranspose(output_E_, output_L_);
    }

    if (bdr_integrals_.size() > 0) {
      if (compatibleWithFaceRestriction(*trial_space_[which])) {
        G_trial_boundary_[which].Gather(input_L_[which], input_E_boundary_[which]);
      }

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
    P_test_->MultTranspose(output_L_, output_T_);

    return output_T_[0];
  }

  /**
   * @brief this function lets the user evaluate the serac::Functional with the given trial space values
   *
   * @param args the input T-vectors
   *
   * note: it accepts exactly `num_trial_spaces` arguments of type mfem::Vector. Additionally, one of those
   * arguments may be a dual_vector, to indicate that Functional::operator() should not only evaluate the
   * element calculations, but also differentiate them w.r.t. the specified dual_vector argument
   */
  template <int wrt, typename... T>
  typename operator_paren_return<wrt>::type operator()(DifferentiateWRT<wrt>, const T&... args)
  {
    const mfem::Vector* input_T[] = {&static_cast<const mfem::Vector&>(args)...};
    static_assert(sizeof...(T) == num_trial_spaces,
                  "Error: Functional::operator() must take exactly as many arguments as trial spaces");

    // get the values for each local processor
    for (uint32_t i = 0; i < num_trial_spaces; i++) {
      P_trial_[i]->Mult(*input_T[i], input_L_[i]);
    }

    output_L_ = 0.0;
    if (domain_integrals_.size() > 0) {
      // get the values for each element on the local processor
      for (uint32_t i = 0; i < num_trial_spaces; i++) {
        G_trial_[i].Gather(input_L_[i], input_E_[i]);
      }

      // compute residual contributions at the element level and sum them
      output_E_ = 0.0;
      for (auto& integral : domain_integrals_) {
        const bool update_state = false;  // QoIs get read-only access to material state
        integral.Mult(input_E_, output_E_, wrt, update_state);
      }

      // scatter-add to compute residuals on the local processor
      G_test_->MultTranspose(output_E_, output_L_);
    }

    if (bdr_integrals_.size() > 0) {
      for (uint32_t i = 0; i < num_trial_spaces; i++) {
        G_trial_boundary_[i].Gather(input_L_[i], input_E_boundary_[i]);
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
      return {output_T_[0], grad_[wrt]};
    }

    if constexpr (wrt == -1) {
      // if the user passes only `mfem::Vector`s then we assume they only want the output value
      //
      // mfem::Vector arg0 = ...;
      // mfem::Vector arg1 = ...;
      // e.g. mfem::Vector value = my_functional(arg0, arg1);
      return output_T_[0];
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
    Gradient(Functional<double(trials...)>& f, uint32_t which = 0)
        : form_(f), which_argument(which), gradient_L_(f.trial_space_[which]->GetVSize())
    {
    }

    void Mult(const mfem::Vector& x, mfem::Vector& y) const { form_.GradientMult(x, y); }

    double operator()(const mfem::Vector& x) const { return form_.ActionOfGradient(x, which_argument); }

    std::unique_ptr<mfem::HypreParVector> assemble()
    {
      // The mfem method ParFiniteElementSpace.NewTrueDofVector should really be marked const
      std::unique_ptr<mfem::HypreParVector> gradient_T(
          const_cast<mfem::ParFiniteElementSpace*>(form_.trial_space_[which_argument])->NewTrueDofVector());

      gradient_L_ = 0.0;

      if (form_.domain_integrals_.size() > 0) {
        auto& K_elem = form_.element_gradients_[which_argument];

        detail::zero_out(K_elem);
        for (auto& domain : form_.domain_integrals_) {
          domain.ComputeElementGradients(view(K_elem), which_argument);
        }

        auto& restriction         = form_.G_trial_[which_argument];
        auto  num_elements        = restriction.num_elements;
        auto  nodes_per_element   = restriction.nodes_per_elem;
        auto  components_per_node = restriction.components;

        for (uint32_t e = 0; e < num_elements; e++) {
          for (uint64_t j = 0; j < nodes_per_element; j++) {
            auto dof = restriction.dof_info(e, j);
            for (uint64_t l = 0; l < components_per_node; l++) {
              int32_t  global_id = int(restriction.GetVDof(dof, l).index());
              uint32_t local_id  = uint32_t(l * nodes_per_element + j);
              gradient_L_(global_id) += dof.sign() * K_elem(e, 0, local_id);
            }
          }
        }
      }

      if (form_.bdr_integrals_.size() > 0) {
        auto& K_belem = form_.bdr_element_gradients_[which_argument];

        detail::zero_out(K_belem);
        for (auto& boundary : form_.bdr_integrals_) {
          boundary.ComputeElementGradients(view(K_belem), which_argument);
        }

        auto& restriction         = form_.G_trial_boundary_[which_argument];
        auto  num_elements        = restriction.num_elements;
        auto  nodes_per_element   = restriction.nodes_per_elem;
        auto  components_per_node = restriction.components;

        for (uint32_t e = 0; e < num_elements; e++) {
          for (uint64_t j = 0; j < nodes_per_element; j++) {
            auto dof = restriction.dof_info(e, j);
            for (uint64_t l = 0; l < components_per_node; l++) {
              int32_t  global_id = int(restriction.GetVDof(dof, l).index());
              uint32_t local_id  = uint32_t(l * nodes_per_element + j);
              gradient_L_(global_id) += dof.sign() * K_belem(e, 0, local_id);
            }
          }
        }
      }

      form_.P_trial_[which_argument]->MultTranspose(gradient_L_, *gradient_T);

      return gradient_T;
    }

    friend auto assemble(Gradient& g) { return g.assemble(); }

  private:
    /**
     * @brief The "parent" @p Functional to calculate gradients with
     */
    Functional<double(trials...), exec>& form_;

    uint32_t which_argument;

    mfem::Vector gradient_L_;
  };

  /// @brief The input set of local DOF values (i.e., on the current rank)
  mutable mfem::Vector input_L_[num_trial_spaces];

  /// @brief The output set of local DOF values (i.e., on the current rank)
  mutable mfem::Vector output_L_;

  /// @brief The input set of per-element DOF values
  mutable std::array<mfem::Vector, num_trial_spaces> input_E_;

  /// @brief The output set of per-element DOF values
  mutable mfem::Vector output_E_;

  /// @brief The input set of per-boundary-element DOF values
  mutable std::array<mfem::Vector, num_trial_spaces> input_E_boundary_;

  /// @brief The output set of per-boundary-element DOF values
  mutable mfem::Vector output_E_boundary_;

  /// @brief The output set of local DOF values (i.e., on the current rank) from boundary elements
  mutable mfem::Vector output_L_boundary_;

  /// @brief The set of true DOF values, used as a scratchpad for @p operator()
  mutable mfem::Vector output_T_;

  /// @brief Manages DOFs for the trial space
  std::array<const mfem::ParFiniteElementSpace*, num_trial_spaces> trial_space_;

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
  ElementRestriction G_trial_[num_trial_spaces];

  /**
   * @brief Operator that converts local (current rank) DOF values to per-boundary element DOF values
   * for the test space
   */
  const mfem::Operator* G_test_boundary_;

  /**
   * @brief Operator that converts local (current rank) DOF values to per-boundary element DOF values
   * for the trial space
   */
  ElementRestriction G_trial_boundary_[num_trial_spaces];

  std::vector<Integral<num_trial_spaces, Q, exec>> integrals_;

  // simplex elements are currently not supported;
  static constexpr mfem::Element::Type supported_types[4] = {mfem::Element::POINT, mfem::Element::SEGMENT,
                                                             mfem::Element::QUADRILATERAL, mfem::Element::HEXAHEDRON};

  /// @brief The objects representing the gradients w.r.t. each input argument of the Functional
  mutable std::vector<Gradient> grad_;

  /// @brief array that stores each element's gradient of the residual w.r.t. trial values
  std::array<ExecArray<double, 3, exec>, num_trial_spaces> element_gradients_;

  /// @brief array that stores each boundary element's gradient of the residual w.r.t. trial values
  std::array<ExecArray<double, 3, exec>, num_trial_spaces> bdr_element_gradients_;
};

}  // namespace serac
