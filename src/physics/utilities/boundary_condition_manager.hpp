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
#include "physics/utilities/solver_config.hpp"

namespace serac {

/**
 * @brief A "view" for filtering a container
 * @note Will be made obsolete by C++20
 * @see std::ranges::views::filter
 */
template <typename Iter, typename Pred>
class FilterView {
public:
  /**
   * @brief An iterator over a filtered view
   */
  class FilterViewIterator {
  public:
    /**
     * @brief Constructs a new iterator object
     * @param[in] curr The element in the container that should be initially "pointed to"
     * @param[in] end The element "one past the end" of the container
     * @param[in] pred The predicate to filter with
     */
    FilterViewIterator(Iter curr, Iter end, const Pred& pred) : curr_(curr), end_(end), pred_(pred) {}

    /**
     * @brief Advances the pointed-to container element to the next element that
     * satisfies the predicate
     */
    FilterViewIterator& operator++()
    {
      // Move forward once to advance the element, then continue
      // advancing until a predicate-satisfying element is found
      ++curr_;
      while ((curr_ != end_) && (!pred_(*curr_))) {
        ++curr_;
      }
      return *this;
    }

    /**
     * @brief Dereferences the iterator
     * @return A non-owning reference to the pointed-to element
     */
    const auto& operator*() const { return *curr_; }

    /**
     * @brief Comparison operation, checks for iterator inequality
     */
    bool operator!=(const FilterViewIterator& other) const { return curr_ != other.curr_; }

  private:
    /**
     * @brief The currently pointed to element
     */
    Iter curr_;

    /**
     * @brief One past the last element of the container, used for bounds checking
     */
    Iter end_;

    /**
     * @brief A reference for the predicate to filter with
     */
    const Pred& pred_;
  };

  /**
   * @brief Constructs a new lazily-evaluated filtering view over a container
   * @param[in] begin The begin() iterator to the container
   * @param[in] end The end() iterator to the container
   * @param[in] pred The predicate for the filter
   */
  FilterView(Iter begin, Iter end, Pred&& pred) : begin_(begin), end_(end), pred_(std::move(pred))
  {
    // Skip to the first element that meets the predicate, making sure not to deref the "end" iterator
    while ((begin_ != end_) && (!pred_(*begin_))) {
      ++begin_;
    }
  }

  /**
   * @brief Returns the first filtered element, i.e., the first element in the
   * underlying container that satisfies the predicate
   */
  FilterViewIterator       begin() { return FilterViewIterator(begin_, end_, pred_); }
  const FilterViewIterator begin() const { return FilterViewIterator(begin_, end_, pred_); }

  /**
   * @brief Returns one past the end of the container, primarily for bounds-checking
   */
  FilterViewIterator       end() { return FilterViewIterator(end_, end_, pred_); }
  const FilterViewIterator end() const { return FilterViewIterator(end_, end_, pred_); }

private:
  /**
   * @brief begin() iterator to the underlying container
   */
  Iter begin_;

  /**
   * @brief end() iterator to the underlying container
   */
  Iter end_;

  /**
   * @brief Predicate to filter with
   */
  Pred pred_;
};

// Deduction guide - iterator and lambda types must be deduced, so
// this mitigates a "builder" function
template <class Iter, class Pred>
FilterView(Iter, Iter, Pred &&) -> FilterView<Iter, Pred>;

template <typename Iter, typename UnaryOp>
class TransformView {
public:
  class TransformViewIterator {
  public:
    TransformViewIterator(Iter curr, const UnaryOp& op) : curr_(curr), op_(op) {}
    TransformViewIterator& operator++()
    {
      ++curr_;
      return *this;
    }
    const auto& operator*() const { return op_(*curr_); }
    auto&       operator*() { return op_(*curr_); }
    bool        operator!=(const TransformViewIterator& other) const { return curr_ != other.curr_; }

  private:
    Iter           curr_;
    const UnaryOp& op_;
  };
  TransformView(Iter begin, Iter end, UnaryOp&& op) : begin_(begin), end_(end), op_(std::move(op)) {}
  TransformViewIterator       begin() { return {begin_, op_}; }
  const TransformViewIterator begin() const { return {begin_, op_}; }
  TransformViewIterator       end() { return {end_, op_}; }
  const TransformViewIterator end() const { return {end_, op_}; }

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
  void addEssential(const std::set<int>& ess_bdr, serac::GeneralCoefficient ess_bdr_coef, const int component = -1)
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
    auto markers = makeMarkers(filtered_attrs);
    bdr_vec.emplace_back(EssentialBoundaryCondition(ess_bdr_coef, component, std::move(markers)));

    // Then mark the boundary attributes as "in use" and invalidate the DOFs cache
    attrs_in_use_[type_key].insert(ess_bdr.begin(), ess_bdr.end());
    all_essential_dofs_valid_[type_key] = false;
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
    auto markers         = makeMarkers(nat_bdr);
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
  void addEssentialTrueDofs(const mfem::Array<int>& true_dofs, serac::GeneralCoefficient ess_bdr_coef,
                            int component = -1)
  {
    using AliasedEssential = StrongAlias<EssentialBoundaryCondition, Tag>;
    EssentialBoundaryCondition bc(ess_bdr_coef, component);
    bc.setTrueDofs(true_dofs);
    std::get<std::vector<AliasedEssential>>(bdrs_).emplace_back(std::move(bc));
    auto type_key                       = typeid(AliasedEssential).hash_code();
    all_essential_dofs_valid_[type_key] = false;
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
  /**
   * @brief Accessor for the generic BC objects
   */
  template <typename BoundaryConditionType>
  auto& generics()
  {
    return std::get<std::vector<BoundaryConditionType>>(bdrs_);
  }

  /**
   * @brief Accessor for the essential BC objects
   */
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
  const auto& generics() const
  {
    return std::get<std::vector<BoundaryConditionType>>(bdrs_);
  }

  mfem::Array<int> makeMarkers(const std::set<int>& attrs) const
  {
    mfem::Array<int> markers(num_attrs_);
    markers = 0;
    for (const int attr : attrs) {
      SLIC_ASSERT_MSG(attr <= num_attrs, "Attribute specified larger than what is found in the mesh.");
      markers[attr - 1] = 1;
    }

    return markers;
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
