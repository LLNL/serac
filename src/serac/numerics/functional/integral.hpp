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

  Integral(std::vector<int> trial_space_indices) : active_trial_spaces(trial_space_indices)
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

  std::vector<int> active_trial_spaces;
  std::vector<int> integral_to_functional_;
  std::vector<int> functional_to_integral_;
};

inline std::array<uint32_t, mfem::Geometry::NUM_GEOMETRIES> geometry_counts(const mfem::Mesh& mesh)
{
  std::array<uint32_t, mfem::Geometry::NUM_GEOMETRIES> counts{};
  for (int i = 0; i < mesh.GetNE(); i++) {
    counts[uint64_t(mesh.GetElementGeometry(i))]++;
  }
  return counts;
}

// note: pyramids and prisms are not currently supported
inline std::vector<mfem::Geometry::Type> supported_geometries_tmp(int dim)
{
  switch (dim) {
    case 1:
      return {mfem::Geometry::SEGMENT};
    case 2:
      return {mfem::Geometry::TRIANGLE, mfem::Geometry::SQUARE};
    case 3:
      return {mfem::Geometry::TETRAHEDRON, mfem::Geometry::CUBE};
    default:
      return {};
  }
}

template <typename signature, int Q, int dim, typename lambda_type, typename qpt_data_type>
Integral MakeDomainIntegral(mfem::Mesh& domain,
                            lambda_type&& qf,  // std::shared_ptr< QuadratureData<qpt_data_type> > qdata,
                            std::vector<int> active_arguments)
{
  Integral integral(active_arguments);

  auto        counts           = geometry_counts(domain);
  std::size_t num_trial_spaces = active_arguments.size();

  for (auto geom : supported_geometries_tmp(dim)) {
    if (counts[uint32_t(geom)] > 0) {
      auto* geom_factors = new serac::GeometricFactors(&domain, Q, geom);
      // auto & X = geom_factors->X;
      // auto & J = geom_factors->J;

      for (std::size_t i = 0; i < num_trial_spaces; i++) {
      }
    }
  }

  return integral;
}

}  // namespace serac
