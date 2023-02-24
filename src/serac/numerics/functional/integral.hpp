#pragma once

#include <array>
#include <memory>

#include "mfem.hpp"

#include "serac/infrastructure/accelerator.hpp"
#include "serac/numerics/functional/geometric_factors.hpp"
#include "serac/numerics/functional/domain_integral_kernels.hpp"

#include "serac/numerics/functional/debug_print.hpp"

namespace serac {

struct Integral {
  enum Type
  {
    Domain,
    Boundary,
    DG,
    _size
  };

  static constexpr std::size_t num_types = Type::_size;

  Integral(Type t, std::vector<uint32_t> trial_space_indices) : type(t), active_trial_spaces(trial_space_indices)
  {
    std::size_t num_trial_spaces = trial_space_indices.size();
    evaluation_with_AD_.resize(num_trial_spaces);
    jvp_.resize(num_trial_spaces);
    element_gradient_.resize(num_trial_spaces);
  }

  void Mult(const std::vector<mfem::BlockVector>& input_E, mfem::BlockVector& output_E, int functional_index,
            bool update_state) const
  {
    int index = (functional_index == -1) ? -1 : functional_to_integral_[static_cast<size_t>(functional_index)];

    auto& kernels = (index == -1) ? evaluation_ : evaluation_with_AD_[uint32_t(index)];
    for (auto& [geometry, func] : kernels) {
      std::vector<const double*> inputs(integral_to_functional_.size());
      for (std::size_t i = 0; i < integral_to_functional_.size(); i++) {
        inputs[i] = input_E[uint32_t(integral_to_functional_[i])].GetBlock(geometry).Read();
      }
      func(inputs, output_E.GetBlock(geometry).ReadWrite(), update_state);
    }
  }

  void GradientMult(const mfem::BlockVector& input_E, mfem::BlockVector& output_E, std::size_t functional_index) const
  {
    int index = functional_to_integral_[functional_index];
    if (index != -1) {
      for (auto& [geometry, func] : jvp_[uint32_t(index)]) {
        func(input_E.GetBlock(geometry).Read(), output_E.GetBlock(geometry).ReadWrite());
      }
    }
  }

  void ComputeElementGradients(std::map<mfem::Geometry::Type, ExecArrayView<double, 3, ExecutionSpace::CPU> >& K_e,
                               std::size_t functional_index) const
  {
    int index = functional_to_integral_[functional_index];
    if (index != -1) {
      for (auto& [geometry, func] : element_gradient_[uint32_t(index)]) {
        func(K_e[geometry]);
      }
    }
  }

  Type type;

  using eval_func = std::function<void(const std::vector<const double*>&, double*, bool)>;
  std::map<mfem::Geometry::Type, eval_func>               evaluation_;
  std::vector<std::map<mfem::Geometry::Type, eval_func> > evaluation_with_AD_;

  using jvp_func = std::function<void(const double*, double*)>;
  std::vector<std::map<mfem::Geometry::Type, jvp_func> > jvp_;

  using grad_func = std::function<void(ExecArrayView<double, 3, ExecutionSpace::CPU>)>;
  std::vector<std::map<mfem::Geometry::Type, grad_func> > element_gradient_;

  std::vector<uint32_t> active_trial_spaces;
  std::vector<int> integral_to_functional_;
  std::vector<int> functional_to_integral_;

  std::map< mfem::Geometry::Type, GeometricFactors > geometric_factors_; 
};

inline std::array<uint32_t, mfem::Geometry::NUM_GEOMETRIES> geometry_counts(const mfem::Mesh& mesh)
{
  std::array<uint32_t, mfem::Geometry::NUM_GEOMETRIES> counts{};
  for (int i = 0; i < mesh.GetNE(); i++) {
    counts[uint64_t(mesh.GetElementGeometry(i))]++;
  }
  return counts;
}

template < typename T >
struct FunctionSignature;

template < typename output_type, typename ... input_types >
struct FunctionSignature< output_type(input_types ... ) > {
  using return_type = output_type;
  using parameter_types = std::tuple< input_types ... >;

  static constexpr int num_args = sizeof ... (input_types);
  static constexpr auto index_seq = std::make_integer_sequence<int, num_args>{};
};

template <typename lambda, int dim, int n, typename... T>
auto batch_apply_qf_no_qdata(lambda qf, const tensor<double, dim, n> x, const T&... inputs)
{
  using return_type = decltype(qf(tensor<double, dim>{}, T{}[0]...));
  tensor<return_type, n> outputs{};
  for (int i = 0; i < n; i++) {
    tensor<double, dim> x_q;
    for (int j = 0; j < dim; j++) {
      x_q[j] = x(j, i);
    }
    outputs[i] = qf(x_q, inputs[i]...);
  }
  return outputs;
}

template <typename lambda, int dim, int n, typename qpt_data_type, typename... T>
auto batch_apply_qf(lambda qf, const tensor<double, dim, n> x, qpt_data_type* qpt_data, bool update_state,
                    const T&... inputs)
{
  using return_type = decltype(qf(tensor<double, dim>{}, qpt_data[0], T{}[0]...));
  tensor<return_type, n> outputs{};
  for (int i = 0; i < n; i++) {
    tensor<double, dim> x_q;
    for (int j = 0; j < dim; j++) {
      x_q[j] = x(j, i);
    }

    auto qdata = qpt_data[i];
    outputs[i] = qf(x_q, qdata, inputs[i]...);
    if (update_state) {
      qpt_data[i] = qdata;
    }
  }
  return outputs;
}

template < int Q, mfem::Geometry::Type geom, typename test, typename ... trials, typename lambda_type, typename state_type, typename derivative_type, int ... indices >
void evaluation_kernel_impl(FunctionSignature< test(trials ...) >,
                            const std::vector<const double*> & inputs, 
                            double * outputs, 
                            const double * positions, 
                            const double * jacobians, 
                            lambda_type qf,
                            QuadratureData< state_type > & qf_state, 
                            derivative_type * qf_derivatives,
                            uint32_t num_elements,
                            bool update_state,
                            std::integer_sequence<int, indices...>)
  {

    static constexpr int differentiation_index = 0;

    using test_element = finite_element<geom, test>;

    /// @brief the element type for each trial space
    static constexpr tuple<finite_element<geom, trials>...> trial_elements{};

    // mfem provides this information as opaque arrays of doubles,
    // so we reinterpret the pointer with
    auto r = reinterpret_cast<typename test_element::dof_type*>(outputs);
    auto x = reinterpret_cast<const typename batched_position<geom, Q>::type*>(positions);
    auto J = reinterpret_cast<const typename batched_jacobian<geom, Q>::type*>(jacobians);
    static constexpr TensorProductQuadratureRule<Q> rule{};

    static constexpr int qpts_per_elem = num_quadrature_points(geom, Q);

    tuple u = {reinterpret_cast<const typename decltype(type<indices>(trial_elements))::dof_type*>(inputs[indices])...};

    // for each element in the domain
    for (uint32_t e = 0; e < num_elements; e++) {

      // load the jacobians and positions for each quadrature point in this element
      auto J_e = J[e];
      auto x_e = x[e];

      // batch-calculate values / derivatives of each trial space, at each quadrature point
      [[maybe_unused]] tuple qf_inputs = {promote_each_to_dual_when<indices == differentiation_index>(
          get<indices>(trial_elements).interpolate(get<indices>(u)[e], rule))...};

      // use J_e to transform values / derivatives on the parent element
      // to the to the corresponding values / derivatives on the physical element
      (parent_to_physical<get<indices>(trial_elements).family>(get<indices>(qf_inputs), J_e), ...);

      // (batch) evalute the q-function at each quadrature point
      //
      // note: the weird immediately-invoked lambda expression is
      // a workaround for a bug in GCC(<12.0) where it fails to
      // decide which function overload to use, and crashes
      auto qf_outputs = [&]() {
        if constexpr (std::is_same_v<state_type, Nothing>) {
          return batch_apply_qf_no_qdata(qf, x_e, get<indices>(qf_inputs)...);
        } else {
          return batch_apply_qf(qf, x_e, qf_state + e * qpts_per_elem, update_state, get<indices>(qf_inputs)...);
        }
      }();

      // use J to transform sources / fluxes on the physical element
      // back to the corresponding sources / fluxes on the parent element
      physical_to_parent<test_element::family>(qf_outputs, J_e);

      // write out the q-function derivatives after applying the
      // physical_to_parent transformation, so that those transformations
      // won't need to be applied in the action_of_gradient and element_gradient kernels
      if constexpr (differentiation_index != -1) {
        for (int q = 0; q < leading_dimension(qf_outputs); q++) {
          qf_derivatives(e, q) = get_gradient(qf_outputs[q]);
        }
      }

      // (batch) integrate the material response against the test-space basis functions
      test_element::integrate(get_value(qf_outputs), rule, &r[e]);
    }
  }

  template <int Q, mfem::Geometry::Type geom, typename signature, typename lambda_type, typename state_type, typename derivative_type >
  std::function<void(const std::vector<const double*>&, double*, bool)> evaluation_kernel(signature s, 
                                                                                          lambda_type qf,
                                                                                          const double* positions,
                                                                                          const double* jacobians,
                                                                                          std::shared_ptr< QuadratureData< state_type > > qf_state, 
                                                                                          std::shared_ptr< derivative_type > qf_derivatives, 
                                                                                          uint32_t num_elements)
  {
    return [=](const std::vector<const double*>& inputs, double* outputs, bool update_state) {
      evaluation_kernel_impl<Q, geom>(s, inputs, outputs, positions, jacobians, qf, *qf_state.get(), qf_derivatives.get(), num_elements,
                                      update_state, s.index_seq);
    };
  }

template < typename lambda_type >
std::function<void(const std::vector<const double*>&, double*, bool)> evaluation_kernel_with_AD(lambda_type) {
  return {};
}

template < typename lambda_type >
std::function<void(const double*, double*)> jvp_kernel(lambda_type) {
  return {};
}

template < typename lambda_type >
std::function<void(ExecArrayView<double, 3, ExecutionSpace::CPU>)> element_gradient_kernel(lambda_type) {
  return {};
}

template <mfem::Geometry::Type geom, int Q, typename signature, typename lambda_type, typename state_type>
void generate_kernels(FunctionSignature<signature> s, Integral& integral, lambda_type&& qf, mfem::Mesh& domain,
                      std::shared_ptr<QuadratureData<state_type> > qf_state)
{
  integral.geometric_factors_[geom] = GeometricFactors(&domain, Q, geom);
  const double * positions = integral.geometric_factors_[geom].X.Read();
  const double * jacobians = integral.geometric_factors_[geom].J.Read();
  const uint32_t num_elements = uint32_t(integral.geometric_factors_[geom].num_elements);

  //void * dummy;
  std::shared_ptr< Nothing > dummy_derivatives;
  integral.evaluation_[geom] = evaluation_kernel< Q, geom >(s, qf, positions, jacobians, qf_state, dummy_derivatives, num_elements);

  //constexpr std::size_t num_args = signature::num_args;
  //for_constexpr<num_args>([&](auto arg) {
  //  integral.evaluation_with_AD_[arg][geom] = evaluation_kernel_with_AD(qf);
  //  integral.jvp_[arg][geom] = jvp_kernel(qf);
  //  integral.element_gradient_[arg][geom] = element_gradient_kernel(qf);
  //});
}

template <typename s, int Q, int dim, typename lambda_type, typename qpt_data_type>
Integral MakeDomainIntegral(mfem::Mesh& domain,
                            lambda_type&& qf,  
                            std::shared_ptr< QuadratureData<qpt_data_type> > qdata,
                            std::vector<uint32_t> argument_indices)
{
  FunctionSignature< s > signature;


  Integral integral(Integral::Type::Domain, argument_indices);

  auto counts = geometry_counts(domain);

  if constexpr (dim == 2) {
    if (counts[uint32_t(mfem::Geometry::TRIANGLE)] > 0) {
      generate_kernels< mfem::Geometry::TRIANGLE, Q >(signature, integral, qf, domain, qdata);
    }
    if (counts[uint32_t(mfem::Geometry::SQUARE)] > 0) {
      generate_kernels< mfem::Geometry::SQUARE, Q >(signature, integral, qf, domain, qdata);
    }
  }

  if constexpr (dim == 3) {
    if (counts[uint32_t(mfem::Geometry::TETRAHEDRON)] > 0) {
      generate_kernels< mfem::Geometry::TETRAHEDRON, Q >(signature, integral, qf, domain, qdata);
    }
    if (counts[uint32_t(mfem::Geometry::CUBE)] > 0) {
      generate_kernels< mfem::Geometry::CUBE, Q >(signature, integral, qf, domain, qdata);
    }
  }

  return integral;
}

}  // namespace serac
