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
struct QoIElementRestriction {
  void ScatterAdd(const mfem::Vector& input, mfem::Vector& output) const { output[0] += input.Sum(); }
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
  Functional(std::array<const mfem::ParFiniteElementSpace*, num_trial_spaces> trial_fes)
      : test_fec_(0, trial_fes[0]->GetMesh()->Dimension()),
        test_space_(dynamic_cast<mfem::ParMesh*>(trial_fes[0]->GetMesh()), &test_fec_),
        trial_space_(trial_fes)
  {
    auto* mesh = trial_fes[0]->GetMesh();

    auto mem_type = mfem::Device::GetMemoryType();

    for (auto type : {Integral::Type::Domain, Integral::Type::Boundary}) {
      input_E_[type].resize(num_trial_spaces);
    }

    for (uint32_t i = 0; i < num_trial_spaces; i++) {
      P_trial_[i] = trial_space_[i]->GetProlongationMatrix();

      input_L_[i].SetSize(P_trial_[i]->Height(), mfem::Device::GetMemoryType());

      for (auto type : {Integral::Type::Domain, Integral::Type::Boundary}) {
        if (type == Integral::Type::Domain) {
          G_trial_[type][i] = BlockElementRestriction(trial_fes[i]);
        } else {
          G_trial_[type][i] = BlockElementRestriction(trial_fes[i], FaceType::BOUNDARY);
        }

        // note: we have to use "Update" here, as mfem::BlockVector's
        // copy assignment ctor (operator=) doesn't let you make changes
        // to the block size
        input_E_[type][i].Update(G_trial_[type][i].bOffsets(), mem_type);
      }
    }

    for (auto type : {Integral::Type::Domain, Integral::Type::Boundary}) {
      std::array<uint32_t, mfem::Geometry::NUM_GEOMETRIES> counts{};
      if (type == Integral::Type::Domain) {
        counts = geometry_counts(*mesh);
      } else {
        counts = boundary_geometry_counts(*mesh);
      }

      mfem::Array<int> offsets(mfem::Geometry::NUM_GEOMETRIES + 1);
      offsets[0] = 0;
      for (int i = 0; i < mfem::Geometry::NUM_GEOMETRIES; i++) {
        auto g         = mfem::Geometry::Type(i);
        offsets[g + 1] = offsets[g] + counts[g];
      }

      output_E_[type].Update(offsets, mem_type);
    }

    G_test_ = QoIElementRestriction();
    P_test_ = new QoIProlongation(trial_fes[0]->GetParMesh()->GetComm());

    output_L_.SetSize(1, mem_type);

    output_T_.SetSize(1, mfem::Device::GetMemoryType());

    // gradient objects depend on some member variables in
    // Functional, so we initialize the gradient objects last
    // to ensure that those member variables are initialized first
    for (uint32_t i = 0; i < num_trial_spaces; i++) {
      grad_.emplace_back(*this, i);
    }
  }

  /// @brief destructor: deallocate the mfem::Operators that we're responsible for
  ~Functional()
  {
    // delete P_test_;
    // delete G_test_;
    // delete G_test_boundary_;
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
    if (domain.GetNE() == 0) return;

    SLIC_ERROR_ROOT_IF(dim != domain.Dimension(), "invalid mesh dimension for domain integral");

    check_for_unsupported_elements(domain);
    check_for_missing_nodal_gridfunc(domain);

    using signature = test(decltype(serac::type<args>(trial_spaces))...);
    integrals_.push_back(
        MakeDomainIntegral<signature, Q, dim>(domain, integrand, qdata, std::vector<uint32_t>{args...}));
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
    auto num_bdr_elements = domain.GetNBE();
    if (num_bdr_elements == 0) return;

    check_for_missing_nodal_gridfunc(domain);

    using signature = test(decltype(serac::type<args>(trial_spaces))...);
    integrals_.push_back(MakeBoundaryIntegral<signature, Q, dim>(domain, integrand, std::vector<uint32_t>{args...}));
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

    // this is used to mark when gather operations have been performed,
    // to avoid doing them more than once per trial space
    bool already_computed[Integral::num_types]{};  // default initializes to `false`

    for (auto& integral : integrals_) {
      auto type = integral.type;

      if (!already_computed[type]) {
        G_trial_[type][which].Gather(input_L_[which], input_E_[type][which]);
        already_computed[type] = true;
      }

      integral.GradientMult(input_E_[type][which], output_E_[type], which);

      // scatter-add to compute residuals on the local processor
      G_test_.ScatterAdd(output_E_[type], output_L_);
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

    // get the values for each local processor
    for (uint32_t i = 0; i < num_trial_spaces; i++) {
      P_trial_[i]->Mult(*input_T[i], input_L_[i]);
    }

    output_L_ = 0.0;

    // this is used to mark when operations have been performed,
    // to avoid doing them more than once
    bool already_computed[Integral::num_types][num_trial_spaces]{};  // default initializes to `false`

    for (auto& integral : integrals_) {
      auto type = integral.type;

      for (auto i : integral.active_trial_spaces) {
        if (!already_computed[type][i]) {
          G_trial_[type][i].Gather(input_L_[i], input_E_[type][i]);
          already_computed[type][i] = true;
        }
      }

      const bool update_state = false;
      integral.Mult(input_E_[type], output_E_[type], wrt, update_state);

      // scatter-add to compute residuals on the local processor
      G_test_.ScatterAdd(output_E_[type], output_L_);
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

      std::map<mfem::Geometry::Type, ExecArray<double, 3, exec>> element_gradients[Integral::num_types];

      for (auto& integral : form_.integrals_) {
        auto& K_elem             = element_gradients[integral.type];
        auto& trial_restrictions = form_.G_trial_[integral.type][which_argument].restrictions;

        if (K_elem.empty()) {
          for (auto& [geom, trial_restriction] : trial_restrictions) {
            K_elem[geom] = ExecArray<double, 3, exec>(trial_restriction.num_elements, 1,
                                                      trial_restriction.nodes_per_elem * trial_restriction.components);

            detail::zero_out(K_elem[geom]);
          }
        }

        integral.ComputeElementGradients(K_elem, which_argument);
      }

      for (auto type : Integral::Types) {
        auto& K_elem             = element_gradients[type];
        auto& trial_restrictions = form_.G_trial_[type][which_argument].restrictions;

        if (!K_elem.empty()) {
          for (auto [geom, elem_matrices] : K_elem) {
            std::vector<DoF> trial_vdofs(
                int(trial_restrictions[geom].nodes_per_elem * trial_restrictions[geom].components));

            for (axom::IndexType e = 0; e < elem_matrices.shape()[0]; e++) {
              trial_restrictions[geom].GetElementVDofs(e, trial_vdofs);

              // note: elem_matrices.shape()[1] is 1 for a QoI
              for (axom::IndexType i = 0; i < elem_matrices.shape()[1]; i++) {
                for (axom::IndexType j = 0; j < elem_matrices.shape()[2]; j++) {
                  int sign = trial_vdofs[j].sign();
                  int col  = int(trial_vdofs[j].index());
                  gradient_L_[col] += sign * elem_matrices(e, i, j);
                }
              }
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

  /// @brief Manages DOFs for the test space
  const mfem::L2_FECollection       test_fec_;
  const mfem::ParFiniteElementSpace test_space_;

  /// @brief Manages DOFs for the trial space
  std::array<const mfem::ParFiniteElementSpace*, num_trial_spaces> trial_space_;

  /**
   * @brief Operator that converts true (global) DOF values to local (current rank) DOF values
   * for the test space
   */
  const mfem::Operator* P_trial_[num_trial_spaces];

  /// @brief The input set of local DOF values (i.e., on the current rank)
  mutable mfem::Vector input_L_[num_trial_spaces];

  BlockElementRestriction G_trial_[Integral::num_types][num_trial_spaces];

  mutable std::vector<mfem::BlockVector> input_E_[Integral::num_types];

  std::vector<Integral> integrals_;

  mutable mfem::BlockVector output_E_[Integral::num_types];

  QoIElementRestriction G_test_;

  /// @brief The output set of local DOF values (i.e., on the current rank)
  mutable mfem::Vector output_L_;

  const mfem::Operator* P_test_;

  /// @brief The set of true DOF values, a reference to this member is returned by @p operator()
  mutable mfem::Vector output_T_;

  /// @brief The objects representing the gradients w.r.t. each input argument of the Functional
  mutable std::vector<Gradient> grad_;
};

}  // namespace serac
