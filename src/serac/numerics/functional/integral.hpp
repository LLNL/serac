#pragma once

#include <array>
#include <memory>

#include "mfem.hpp"

#include "serac/infrastructure/accelerator.hpp"
#include "serac/numerics/functional/geometric_factors.hpp"
#include "serac/numerics/functional/function_signature.hpp"
#include "serac/numerics/functional/domain_integral_kernels.hpp"
#include "serac/numerics/functional/boundary_integral_kernels.hpp"

#include "serac/numerics/functional/debug_print.hpp"

namespace serac {

struct Integral {
  enum Type
  {
    Domain,
    Boundary,
    // DG, unimplemented
    _size
  };

  static constexpr std::size_t num_types = Type::_size;

  Integral(Type t, std::vector<uint32_t> trial_space_indices) : type(t), active_trial_spaces(trial_space_indices)
  {
    std::size_t num_trial_spaces = trial_space_indices.size();
    evaluation_with_AD_.resize(num_trial_spaces);
    jvp_.resize(num_trial_spaces);
    element_gradient_.resize(num_trial_spaces);

    functional_to_integral_ = std::vector<int>(num_trial_spaces, -1);
    for (size_t i = 0; i < num_trial_spaces; i++) {
      functional_to_integral_[active_trial_spaces[i]] = static_cast<int>(i);
    }
  }

  void Mult(const std::vector<mfem::BlockVector>& input_E, mfem::BlockVector& output_E, int functional_index,
            bool update_state) const
  {
    int index = (functional_index == -1) ? -1 : functional_to_integral_[static_cast<size_t>(functional_index)];

    output_E = 0.0;

    auto& kernels = (index == -1) ? evaluation_ : evaluation_with_AD_[uint32_t(index)];
    for (auto& [geometry, func] : kernels) {
      std::vector<const double*> inputs(active_trial_spaces.size());
      for (std::size_t i = 0; i < active_trial_spaces.size(); i++) {
        inputs[i] = input_E[uint32_t(active_trial_spaces[i])].GetBlock(geometry).Read();
      }
      func(inputs, output_E.GetBlock(geometry).ReadWrite(), update_state);
    }
  }

  void GradientMult(const mfem::BlockVector& input_E, mfem::BlockVector& output_E, std::size_t functional_index) const
  {
    int index = functional_to_integral_[functional_index];
    if (index != -1) {
      output_E = 0.0;
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
  std::vector<int> functional_to_integral_;

  std::map< mfem::Geometry::Type, GeometricFactors > geometric_factors_; 
};


template <mfem::Geometry::Type geom, int Q, typename test, typename ... trials, typename lambda_type, typename state_type>
void generate_kernels(FunctionSignature<test(trials...)> s, Integral& integral, lambda_type&& qf, mfem::Mesh& domain,
                      std::shared_ptr<QuadratureData<state_type> > qf_state)
{
  integral.geometric_factors_[geom] = GeometricFactors(&domain, Q, geom);
  GeometricFactors & gf = integral.geometric_factors_[geom];
  if (gf.num_elements == 0) return;

  const double * positions = gf.X.Read();
  const double * jacobians = gf.J.Read();
  const uint32_t num_elements = uint32_t(gf.num_elements);
  const uint32_t qpts_per_element = num_quadrature_points(geom, Q);

  constexpr int         NO_DIFFERENTIATION = -1;
  std::shared_ptr<zero> dummy_derivatives;
  integral.evaluation_[geom] = domain_integral::evaluation_kernel<NO_DIFFERENTIATION, Q, geom>(
      s, qf, positions, jacobians, qf_state, dummy_derivatives, num_elements);

  constexpr std::size_t num_args = s.num_args;
  [[maybe_unused]] static constexpr int dim = dimension_of(geom);
  for_constexpr<num_args>([&](auto index) {

    // allocate memory for the derivatives of the q-function at each quadrature point
    //
    // Note: ptrs' lifetime is managed in an unusual way! It is captured by-value in the
    // action_of_gradient functor below to augment the reference count, and extend its lifetime to match
    // that of the DomainIntegral that allocated it.
    using derivative_type   = decltype(domain_integral::get_derivative_type<index, dim, trials...>(qf, state_type{}));
    auto ptr = accelerator::make_shared_array<ExecutionSpace::CPU, derivative_type>(num_elements * qpts_per_element);

    integral.evaluation_with_AD_[index][geom] = domain_integral::evaluation_kernel<index, Q, geom>(
      s, qf, positions, jacobians, qf_state, ptr, num_elements);
    
    integral.jvp_[index][geom] = domain_integral::jvp_kernel<index, Q, geom>(s, jacobians, ptr, num_elements);
    integral.element_gradient_[index][geom] = domain_integral::element_gradient_kernel(qf);
  });
}

template <typename s, int Q, int dim, typename lambda_type, typename qpt_data_type>
Integral MakeDomainIntegral(mfem::Mesh& domain,
                            lambda_type&& qf,  
                            std::shared_ptr< QuadratureData<qpt_data_type> > qdata,
                            std::vector<uint32_t> argument_indices)
{
  FunctionSignature<s> signature;

  Integral integral(Integral::Type::Domain, argument_indices);

  if constexpr (dim == 2) {
    generate_kernels<mfem::Geometry::TRIANGLE, Q>(signature, integral, qf, domain, qdata);
    generate_kernels<mfem::Geometry::SQUARE, Q>(signature, integral, qf, domain, qdata);
  }

  if constexpr (dim == 3) {
    //generate_kernels<mfem::Geometry::TETRAHEDRON, Q>(signature, integral, qf, domain, qdata);
    generate_kernels<mfem::Geometry::CUBE, Q>(signature, integral, qf, domain, qdata);
  }

  return integral;
}

template <mfem::Geometry::Type geom, int Q, typename test, typename ... trials, typename lambda_type>
void generate_bdr_kernels(FunctionSignature<test(trials...)> s, Integral& integral, lambda_type&& qf, mfem::Mesh& domain)
{
  integral.geometric_factors_[geom] = GeometricFactors(&domain, Q, geom, FaceType::BOUNDARY);
  GeometricFactors & gf = integral.geometric_factors_[geom];
  if (gf.num_elements == 0) return;

  const double * positions = gf.X.Read();
  const double * jacobians = gf.J.Read();
  const uint32_t num_elements = uint32_t(gf.num_elements);
  const uint32_t qpts_per_element = num_quadrature_points(geom, Q);

  constexpr int         NO_DIFFERENTIATION = -1;
  std::shared_ptr<zero> dummy_derivatives;
  integral.evaluation_[geom] = boundary_integral::evaluation_kernel<NO_DIFFERENTIATION, Q, geom>(
      s, qf, positions, jacobians, dummy_derivatives, num_elements);

  constexpr std::size_t num_args = s.num_args;
  [[maybe_unused]] static constexpr int dim = dimension_of(geom);
  for_constexpr<num_args>([&](auto index) {

    // allocate memory for the derivatives of the q-function at each quadrature point
    //
    // Note: ptrs' lifetime is managed in an unusual way! It is captured by-value in the
    // action_of_gradient functor below to augment the reference count, and extend its lifetime to match
    // that of the boundaryIntegral that allocated it.
    using derivative_type   = decltype(boundary_integral::get_derivative_type<index, dim, trials...>(qf));
    auto ptr = accelerator::make_shared_array<ExecutionSpace::CPU, derivative_type>(num_elements * qpts_per_element);

    integral.evaluation_with_AD_[index][geom] = boundary_integral::evaluation_kernel<index, Q, geom>(
      s, qf, positions, jacobians, ptr, num_elements);
    
    integral.jvp_[index][geom] = boundary_integral::jvp_kernel<index, Q, geom>(s, jacobians, ptr, num_elements);
    integral.element_gradient_[index][geom] = boundary_integral::element_gradient_kernel(qf);
  });
}

template <typename s, int Q, int dim, typename lambda_type>
Integral MakeBoundaryIntegral(mfem::Mesh& domain,
                              lambda_type&& qf,  
                              std::vector<uint32_t> argument_indices)
{
  FunctionSignature<s> signature;

  Integral integral(Integral::Type::Boundary, argument_indices);

  if constexpr (dim == 1) {
    generate_bdr_kernels<mfem::Geometry::SEGMENT, Q>(signature, integral, qf, domain);
  }

  if constexpr (dim == 2) {
    generate_bdr_kernels<mfem::Geometry::TRIANGLE, Q>(signature, integral, qf, domain);
    generate_bdr_kernels<mfem::Geometry::SQUARE, Q>(signature, integral, qf, domain);
  }

  return integral;
}

}  // namespace serac
