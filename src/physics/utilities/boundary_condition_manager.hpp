// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file boundary_condition_manager.hpp
 *
 * @brief This file contains the declaration of the boundary condition manager class
 */

#ifndef BOUNDARY_CONDITION_MANAGER
#define BOUNDARY_CONDITION_MANAGER

#include <memory>
#include <set>
#include <typeinfo>
#include <unordered_map>

#include "physics/utilities/boundary_condition.hpp"
#include "physics/utilities/finite_element_state.hpp"

namespace serac {

/**
 * @brief A "view" for lazily transforming a container
 * @note Will be made obsolete by C++20
 * @see std::ranges::views::transform
 * @tparam The iterator type (must satisfy ForwardIterator)
 * @tparam UnaryOp The operator to transform with
 */
template <typename Iter, typename UnaryOp>
class TransformView {
public:
  class TransformViewIterator {
  public:
    TransformViewIterator(Iter curr, const UnaryOp& op) : curr_(curr), op_(op) {}
    /**
     * @brief Advances the iterator to the next element of the view
     */
    TransformViewIterator& operator++()
    {
      ++curr_;
      return *this;
    }
    /**
     * @brief Dereferences the iterator
     */
    const auto& operator*() const { return op_(*curr_); }
    auto&       operator*() { return op_(*curr_); }
    /**
     * @brief Iterator comparison
     */
    bool operator!=(const TransformViewIterator& other) const { return curr_ != other.curr_; }

  private:
    Iter           curr_;
    const UnaryOp& op_;
  };
  TransformView(Iter begin, Iter end, UnaryOp&& op) : begin_(begin), end_(end), op_(std::move(op)) {}
  /**
   * @brief Returns an iterator to the first element of the view
   */
  TransformViewIterator       begin() { return {begin_, op_}; }
  const TransformViewIterator begin() const { return {begin_, op_}; }
  /**
   * @brief Returns an iterator to one past the end of the view
   */
  TransformViewIterator       end() { return {end_, op_}; }
  const TransformViewIterator end() const { return {end_, op_}; }

  /**
   * @brief Returns the size of the view
   */
  auto size() const { return std::distance(begin_, end_); }

private:
  Iter    begin_;
  Iter    end_;
  UnaryOp op_;
};

/**
 * @brief Wrapper to enable strong type aliases
 * i.e. those where different aliases of the same type are not interchangeable
 * Roughly equivalent to the following Ada:
 * @code{.ada}
 * type Alias is new AliasedType;
 * @endcode
 * This should be used as follows:
 * @code{.cpp}
 * using Meters = StrongAlias<double, struct MetersParam>;
 * @endcode
 * @tparam AliasedType The type to tag/wrap/alias
 * @tparam Alias The type to use as a tag - typically an empty struct
 */
template <class AliasedType, class Alias>
class StrongAlias {
public:
  /**
   * @brief Constructs a new StrongAlias instance via move from rvalue
   * @param[in] val The value to move from
   */
  explicit StrongAlias(AliasedType&& val) : val_(std::move(val)) {}
  /**
   * @brief Constructs a new StrongAlias instance via copy from lvalue
   * @param[in] val The value to copy from
   */
  explicit StrongAlias(const AliasedType& val) : val_(val) {}

  /**
   * @brief Accesses the underlying AliasedType object
   */
  explicit operator const AliasedType&() const { return val_; }
  explicit operator AliasedType&() { return val_; }
  explicit operator AliasedType&&() && { return std::move(val_); }

private:
  AliasedType val_;
};

/**
 * @brief Type trait for determining whether a type has a static bool member called should_be_scalar
 */
template <typename T, typename = void>
struct has_should_be_scalar : std::false_type {
};

template <typename T>
struct has_should_be_scalar<T,
                            std::enable_if_t<std::is_same_v<bool, std::decay_t<decltype(T::should_be_scalar)>>, void>>
    : std::true_type {
};

/**
 * @brief A compile-time check called when an essential boundary condition is instantiated
 * with a scalar coefficient but no component
 * @tparam Tag the physics-specific tag type for the BC
 */
template <typename Tag>
constexpr void check_scalar_coef_with_no_component()
{
  // If this is a vector valued boundary condition and a scalar was passed,
  // there must be a component associated with it
  if constexpr (has_should_be_scalar<Tag>::value) {
    if constexpr (!Tag::should_be_scalar) {
      static_assert(sizeof(Tag) < 0, "A component is required!");
    }
  }
  // If the type of the BC is not known, request that the user pass
  // a component
  else {
    static_assert(
        sizeof(Tag) < 0,
        "Boundary condition does not specify whether it is scalar- or vector-valued.  If this is a scalar-valued BC, "
        "pass a component of zero, otherwise, pass the vector component to which this scalar coef should apply.");
  }
}

/**
 * @brief A structure for storing boundary conditions of arbitrary type
 * @tparam BoundaryConditionTypes The variadic list of types
 */
template <class... BoundaryConditionTypes>
class BoundaryConditionManager {
public:
  BoundaryConditionManager(const mfem::ParMesh& mesh) : num_attrs_(mesh.bdr_attributes.Max()) {}

  /**
   * @brief Set the essential boundary conditions from a list of boundary markers and a coefficient
   *
   * @param[in] ess_bdr The set of essential BC attributes
   * @param[in] ess_bdr_coef The essential BC value coefficient
   * @param[in] component The component to set (-1 implies all components are set)
   * @tparam Tag The physics-specific tag/marker type to use to annotate the essential BC
   */
  template <typename Tag>
  void addEssential(const std::set<int>& ess_bdr, serac::GeneralCoefficient ess_bdr_coef, const int component)
  {
    using AliasedEssential = StrongAlias<EssentialBoundaryCondition, Tag>;
    auto          type_key = typeid(AliasedEssential).hash_code();
    auto&         bdr_vec  = std::get<std::vector<AliasedEssential>>(bdrs_);
    std::set<int> filtered_attrs;
    std::set_difference(ess_bdr.begin(), ess_bdr.end(), attrs_in_use_[type_key].begin(), attrs_in_use_[type_key].end(),
                        std::inserter(filtered_attrs, filtered_attrs.begin()));

    // Check if anything was removed
    if (filtered_attrs.size() < bdr_vec.size()) {
      SLIC_WARNING("Multiple definition of essential boundary! Using first definition given.");
    }

    // Build the markers and then the boundary condition
    auto markers = BoundaryCondition::makeMarkers(filtered_attrs, num_attrs_);
    bdr_vec.emplace_back(EssentialBoundaryCondition(ess_bdr_coef, component, std::move(markers)));

    // Then mark the boundary attributes as "in use" and invalidate the DOFs cache
    attrs_in_use_[type_key].insert(ess_bdr.begin(), ess_bdr.end());
    all_essential_dofs_valid_[type_key] = false;
  }

  /**
   * @brief Checks that a scalar coefficient without a specified component is valid
   */
  template <typename Tag>
  void addEssential(const std::set<int>& ess_bdr, std::shared_ptr<mfem::Coefficient> ess_bdr_coef)
  {
    check_scalar_coef_with_no_component<Tag>();
    // If a scalar is acceptable, use component zero (element of single-element vector)
    addEssential<Tag>(ess_bdr, ess_bdr_coef, 0);
  }

  /**
   * @brief Defaults coefficient component to -1 for vector coefficients only
   */
  template <typename Tag>
  void addEssential(const std::set<int>& ess_bdr, std::shared_ptr<mfem::VectorCoefficient> ess_bdr_coef)
  {
    // Vector coefficient applies to all components
    addEssential<Tag>(ess_bdr, ess_bdr_coef, -1);
  }

  /**
   * @brief Set the natural boundary conditions from a list of boundary markers and a coefficient
   *
   * @param[in] nat_bdr The set of mesh attributes denoting a natural boundary
   * @param[in] nat_bdr_coef The coefficient defining the natural boundary function
   * @param[in] component The component to set (-1 implies all components are set)
   * @tparam Tag The physics-specific tag/marker type to use to annotate the essential BC
   */
  template <typename Tag>
  void addNatural(const std::set<int>& nat_bdr, serac::GeneralCoefficient nat_bdr_coef, const int component = -1)
  {
    using AliasedNatural = StrongAlias<NaturalBoundaryCondition, Tag>;
    auto markers         = BoundaryCondition::makeMarkers(nat_bdr, num_attrs_);
    std::get<std::vector<AliasedNatural>>(bdrs_).emplace_back(
        NaturalBoundaryCondition(nat_bdr_coef, component, std::move(markers)));
  }

  /**
   * @brief Set a generic boundary condition from a list of boundary markers and a coefficient
   *
   * @param[in] bdr_attr The set of mesh attributes denoting a natural boundary
   * @param[in] bdr_coef The coefficient defining the natural boundary function
   * @param[in] component The component to set (-1 implies all components are set)
   * @tparam BoundaryConditionType The type of the BC to add
   */
  template <typename BoundaryConditionType>
  void addGeneric(BoundaryConditionType&& bc)
  {
    std::get<BoundaryConditionType>(bdrs_).emplace_back(std::move(bc));
  }

  /**
   * @brief Set a list of true degrees of freedom from a coefficient
   *
   * @param[in] true_dofs The true degrees of freedom to set with a Dirichlet condition
   * @param[in] ess_bdr_coef The coefficient that evaluates to the Dirichlet condition
   * @param[in] component The component to set (-1 implies all components are set)
   * @tparam Tag The physics-specific tag/marker type to use to annotate the essential BC
   */
  template <typename Tag>
  void addEssentialTrueDofs(const mfem::Array<int>& true_dofs, serac::GeneralCoefficient ess_bdr_coef, int component)
  {
    using AliasedEssential = StrongAlias<EssentialBoundaryCondition, Tag>;
    EssentialBoundaryCondition bc(ess_bdr_coef, component);
    bc.setTrueDofs(true_dofs);
    std::get<std::vector<AliasedEssential>>(bdrs_).emplace_back(std::move(bc));
    auto type_key                       = typeid(AliasedEssential).hash_code();
    all_essential_dofs_valid_[type_key] = false;
  }

  /**
   * @brief Checks that a scalar coefficient without a specified component is valid
   */
  template <typename Tag>
  void addEssentialTrueDofs(const mfem::Array<int>& true_dofs, std::shared_ptr<mfem::Coefficient> ess_bdr_coef)
  {
    check_scalar_coef_with_no_component<Tag>();
    // If a scalar is acceptable, use component zero (element of single-element vector)
    addEssentialTrueDofs<Tag>(true_dofs, ess_bdr_coef, 0);
  }

  /**
   * @brief Defaults coefficient component to -1 for vector coefficients only
   */
  template <typename Tag>
  void addEssentialTrueDofs(const mfem::Array<int>& true_dofs, std::shared_ptr<mfem::VectorCoefficient> ess_bdr_coef)
  {
    // Vector coefficient applies to all components
    addEssentialTrueDofs<Tag>(true_dofs, ess_bdr_coef, -1);
  }

  /**
   * @brief Returns all the degrees of freedom associated with all the essential BCs
   * @return A const reference to the list of DOF indices, without duplicates and sorted
   * @tparam Tag The physics-specific tag/marker type to use to annotate the essential BC
   */
  template <typename Tag>
  const mfem::Array<int>& allEssentialDofs() const
  {
    using AliasedEssential = StrongAlias<EssentialBoundaryCondition, Tag>;
    auto type_key          = typeid(AliasedEssential).hash_code();
    if (!all_essential_dofs_valid_[type_key]) {
      updateAllEssentialDofs<AliasedEssential>();
    }
    return all_essential_dofs_[type_key];
  }

  /**
   * @brief Eliminates all essential BCs from a matrix
   * @param[inout] matrix The matrix to eliminate from, will be modified
   * @tparam Tag The physics-specific tag/marker type to use to annotate the essential BC
   * @return The eliminated matrix entries
   * @note The sum of the eliminated matrix and the modified parameter is
   * equal to the initial state of the parameter
   */
  template <typename Tag>
  std::unique_ptr<mfem::HypreParMatrix> eliminateAllEssentialDofsFromMatrix(mfem::HypreParMatrix& matrix) const
  {
    return std::unique_ptr<mfem::HypreParMatrix>(matrix.EliminateRowsCols(allEssentialDofs<Tag>()));
  }

  /**
   * @brief Accessor for the essential BC objects
   */
  template <typename Tag>
  auto essentials()
  {
    auto& vec = std::get<std::vector<StrongAlias<EssentialBoundaryCondition, Tag>>>(bdrs_);
    return TransformView(vec.begin(), vec.end(), [](auto& bc) -> EssentialBoundaryCondition& {
      return static_cast<EssentialBoundaryCondition&>(bc);
    });
  }
  template <typename Tag>
  const auto essentials() const
  {
    const auto& vec = std::get<std::vector<StrongAlias<EssentialBoundaryCondition, Tag>>>(bdrs_);
    return TransformView(vec.begin(), vec.end(), [](const auto& bc) -> const EssentialBoundaryCondition& {
      return static_cast<const EssentialBoundaryCondition&>(bc);
    });
  }
  /**
   * @brief Accessor for the natural BC objects
   */
  template <typename Tag>
  auto naturals()
  {
    auto& vec = std::get<std::vector<StrongAlias<NaturalBoundaryCondition, Tag>>>(bdrs_);
    return TransformView(vec.begin(), vec.end(), [](auto& bc) -> NaturalBoundaryCondition& {
      return static_cast<NaturalBoundaryCondition&>(bc);
    });
  }
  template <typename Tag>
  const auto naturals() const
  {
    const auto& vec = std::get<std::vector<StrongAlias<NaturalBoundaryCondition, Tag>>>(bdrs_);
    return TransformView(vec.begin(), vec.end(), [](const auto& bc) -> const NaturalBoundaryCondition& {
      return static_cast<const NaturalBoundaryCondition&>(bc);
    });
  }

  /**
   * @brief Accessor for the generic BC objects
   */
  template <typename BoundaryConditionType>
  auto& generics()
  {
    return std::get<std::vector<BoundaryConditionType>>(bdrs_);
  }
  template <typename BoundaryConditionType>
  const auto& generics() const
  {
    return std::get<std::vector<BoundaryConditionType>>(bdrs_);
  }

private:
  /**
   * @brief Updates the "cached" list of all DOF indices
   * @tparam AliasedEssential The physics-specific strongly aliased essential BC type
   */
  template <typename AliasedEssential>
  void updateAllEssentialDofs() const
  {
    auto type_key = typeid(AliasedEssential).hash_code();
    all_essential_dofs_[type_key].DeleteAll();
    for (const auto& bc : std::get<std::vector<AliasedEssential>>(bdrs_)) {
      all_essential_dofs_[type_key].Append(static_cast<const EssentialBoundaryCondition&>(bc).getTrueDofs());
    }
    all_essential_dofs_[type_key].Sort();
    all_essential_dofs_[type_key].Unique();
    all_essential_dofs_valid_[type_key] = true;
  }

  /**
   * @brief The total number of boundary attributes for a mesh
   */
  const int num_attrs_;

  /**
   * @brief The vector of boundary conditions
   */
  std::tuple<std::vector<BoundaryConditionTypes>...> bdrs_;

  /**
   * @brief The set of boundary attributes associated with
   * already-registered BCs
   * @see https://mfem.org/mesh-formats/
   */
  std::unordered_map<std::size_t, std::set<int>> attrs_in_use_;

  /**
   * @brief The set of true DOF indices corresponding
   * to all registered BCs
   */
  mutable std::unordered_map<std::size_t, mfem::Array<int>> all_essential_dofs_;

  /**
   * @brief Whether the set of stored total DOFs is valid
   */
  mutable std::unordered_map<std::size_t, bool> all_essential_dofs_valid_;
};

}  // namespace serac

#endif
