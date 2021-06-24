// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file finite_element_state.hpp
 *
 * @brief This file contains the declaration of structure that manages the MFEM objects
 * that make up the state for a given field
 */

#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <type_traits>
#include <variant>

#include "mfem.hpp"

namespace serac {

namespace detail {

template <typename T0, typename T1, typename SFINAE = void>
struct variant_storage {
  int index_ = 0;
  union {
    T0 t0_;
    T1 t1_;
  };

  constexpr variant_storage(const variant_storage& other) : index_(other.index_)
  {
    switch (index_) {
      case 0: {
        new (&t0_) T0(other.t0_);
        break;
      }
      case 1: {
        new (&t1_) T1(other.t1_);
        break;
      }
    }
  };

  constexpr variant_storage(variant_storage&& other) : index_(other.index_)
  {
    switch (index_) {
      case 0: {
        new (&t0_) T0(std::move(other.t0_));
        break;
      }
      case 1: {
        new (&t1_) T1(std::move(other.t1_));
        break;
      }
    }
  };
  constexpr void clear()
  {
    switch (index_) {
      case 0: {
        t0_.~T0();
        break;
      }
      case 1: {
        t1_.~T1();
        break;
      }
    }
  }

  constexpr variant_storage() : index_{0}, t0_{} {}
  ~variant_storage() { clear(); }
};

template <typename T0, typename T1>
struct variant_storage<T0, T1,
                       std::enable_if_t<std::is_trivially_destructible_v<T0> && std::is_trivially_destructible_v<T1>>> {
  int index_ = 0;
  union {
    T0 t0_;
    T1 t1_;
  };
  constexpr variant_storage() : index_{0}, t0_{} {}
  constexpr void clear() {}
};

template <typename T, typename T0, typename T1>
struct is_variant_assignable {
  constexpr static bool value = std::is_same_v<std::decay_t<T>, T0> || std::is_assignable_v<T0, T> ||
                                std::is_same_v<std::decay_t<T>, T1> || std::is_assignable_v<T1, T>;
};

}  // namespace detail

// Should we #include <variant> for std::variant_alternative??
template <int I, typename T0, typename T1>
struct variant_alternative;

template <typename T0, typename T1>
struct variant_alternative<0, T0, T1> {
  using type = T0;
};

template <typename T0, typename T1>
struct variant_alternative<1, T0, T1> {
  using type = T1;
};

template <typename T0, typename T1>
struct variant {
  detail::variant_storage<T0, T1> storage_;
  constexpr variant() = default;

  constexpr variant(const variant& other) = default;
  constexpr variant(variant&& other)      = default;

  template <typename T, typename SFINAE = std::enable_if_t<detail::is_variant_assignable<T, T0, T1>::value>>
  constexpr variant(T&& t)
  {
    if constexpr (std::is_same_v<std::decay_t<T>, T0> || std::is_assignable_v<T0, T>) {
      storage_.index_ = 0;
      new (&storage_.t0_) T0(std::forward<T>(t));
    } else if constexpr (std::is_same_v<std::decay_t<T>, T1> || std::is_assignable_v<T1, T>) {
      storage_.index_ = 1;
      new (&storage_.t1_) T1(std::forward<T>(t));
    } else {
      static_assert(sizeof(T) < 0, "Type not supported");
    }
  }

  constexpr variant& operator=(const variant& other) = default;
  constexpr variant& operator=(variant&& other) = default;

  template <typename T, typename SFINAE = std::enable_if_t<detail::is_variant_assignable<T, T0, T1>::value>>
  constexpr variant& operator=(T&& t)
  {
    // FIXME: Things that are convertible to T0 etc
    if constexpr (std::is_same_v<std::decay_t<T>, T0>) {
      if (storage_.index_ != 0) {
        storage_.clear();
      }
      storage_.t0_    = std::forward<T>(t);
      storage_.index_ = 0;
    } else if constexpr (std::is_same_v<std::decay_t<T>, T1>) {
      if (storage_.index_ != 1) {
        storage_.clear();
      }
      storage_.t1_    = std::forward<T>(t);
      storage_.index_ = 1;
    } else {
      static_assert(sizeof(T) < 0, "Type not supported");
    }
    return *this;
  }

  constexpr int index() const { return storage_.index_; }

  template <int I>
  friend constexpr typename variant_alternative<I, T0, T1>::type& get(variant& v)
  {
    if constexpr (I == 0) {
      return v.storage_.t0_;
    } else if constexpr (I == 1) {
      return v.storage_.t1_;
    }
  }

  template <int I>
  friend constexpr const typename variant_alternative<I, T0, T1>::type& get(const variant& v)
  {
    if constexpr (I == 0) {
      return v.storage_.t0_;
    } else if constexpr (I == 1) {
      return v.storage_.t1_;
    }
  }
};

template <typename T, typename T0, typename T1>
constexpr T& get(variant<T0, T1>& v)
{
  if constexpr (std::is_same_v<T, T0>) {
    return get<0>(v);
  } else if constexpr (std::is_same_v<T, T1>) {
    return get<1>(v);
  }
}

template <typename T, typename T0, typename T1>
constexpr const T& get(const variant<T0, T1>& v)
{
  if constexpr (std::is_same_v<T, T0>) {
    return get<0>(v);
  } else if constexpr (std::is_same_v<T, T1>) {
    return get<1>(v);
  }
}

template <typename Visitor, typename Variant>
constexpr decltype(std::declval<Visitor&>()(get<0>(std::declval<Variant&>()))) visit(Visitor visitor, Variant&& v)
{
  if (v.index() == 0) {
    return visitor(get<0>(v));
  } else {
    return visitor(get<1>(v));
  }
}

template <typename T, typename T0, typename T1>
bool holds_alternative(const variant<T0, T1>& v)
{
  if constexpr (std::is_same_v<T, T0>) {
    return v.index() == 0;
  } else if constexpr (std::is_same_v<T, T1>) {
    return v.index() == 1;
  }
  return false;
}

template <typename T, typename T0, typename T1>
T* get_if(variant<T0, T1>* v)
{
  if constexpr (std::is_same_v<T, T0>) {
    return (v->index() == 0) ? &get<0>(*v) : nullptr;
  } else if constexpr (std::is_same_v<T, T1>) {
    return (v->index() == 1) ? &get<1>(*v) : nullptr;
  }
  return nullptr;
}

template <typename T, typename T0, typename T1>
const T* get_if(const variant<T0, T1>* v)
{
  if constexpr (std::is_same_v<T, T0>) {
    return (v->index() == 0) ? &get<0>(*v) : nullptr;
  } else if constexpr (std::is_same_v<T, T1>) {
    return (v->index() == 1) ? &get<1>(*v) : nullptr;
  }
  return nullptr;
}

namespace detail {

/**
 * @brief A helper type for uniform semantics over owning/non-owning pointers
 *
 * This logic is needed to integrate with the mesh and field reconstruction logic
 * provided by Sidre's MFEMSidreDataCollection.  When a Serac restart occurs, the
 * saved data is used to construct fully functional mfem::(Par)Mesh and
 * mfem::(Par)GridFunction objects.  The FiniteElementCollection and (Par)FiniteElementSpace
 * objects are intermediates in the construction of these objects and are therefore owned
 * by the MFEMSidreDataCollection in the case of a restart/reconstruction.  In a normal run,
 * Serac constructs the mesh and fields, so these FEColl and FESpace objects are owned
 * by Serac.  In both cases, the MFEMSidreDataCollection maintains ownership of the mesh
 * and field objects themselves.
 */
template <typename T>
using MaybeOwningPointer = variant<T*, std::unique_ptr<T>>;

/**
 * @brief Retrieves a reference to the underlying object in a MaybeOwningPointer
 * @param[in] obj The object to dereference
 */
template <typename T>
static T& retrieve(MaybeOwningPointer<T>& obj)
{
  return visit([](auto&& ptr) -> T& { return *ptr; }, obj);
}
/// @overload
template <typename T>
static const T& retrieve(const MaybeOwningPointer<T>& obj)
{
  return visit([](auto&& ptr) -> const T& { return *ptr; }, obj);
}

}  // namespace detail

/**
 * @brief A sum type for encapsulating either a scalar or vector coeffient
 */
using GeneralCoefficient = variant<std::shared_ptr<mfem::Coefficient>, std::shared_ptr<mfem::VectorCoefficient>>;

/**
 * @brief convenience function for querying the type stored in a GeneralCoefficient
 */
inline bool is_scalar_valued(const GeneralCoefficient& coef)
{
  return holds_alternative<std::shared_ptr<mfem::Coefficient>>(coef);
}

/**
 * @brief convenience function for querying the type stored in a GeneralCoefficient
 */
inline bool is_vector_valued(const GeneralCoefficient& coef)
{
  return holds_alternative<std::shared_ptr<mfem::VectorCoefficient>>(coef);
}

/**
 * @brief Class for encapsulating the critical MFEM components of a solver
 *
 * Namely: Mesh, FiniteElementCollection, FiniteElementState,
 * GridFunction, and a Vector of the solution
 */
class FiniteElementState {
public:
  /**
   * @brief Structure for optionally configuring a FiniteElementState
   */
  // The optionals are explicitly default-constructed to allow the user to partially aggregrate-initialized
  // with only the options they care about
  struct Options {
    /**
     * @brief The polynomial order that should be used for the problem
     */
    int order = 1;
    /**
     * @brief The number of copies of the finite element collections (e.g. vector_dim = 2 or 3 for solid mechanics).
     * Defaults to scalar valued spaces.
     */
    int vector_dim = 1;
    /**
     * @brief The FECollection to use - defaults to an H1_FECollection
     */
    std::unique_ptr<mfem::FiniteElementCollection> coll = {};
    /**
     * The DOF ordering that should be used interally by MFEM
     */
    mfem::Ordering::Type ordering = mfem::Ordering::byVDIM;
    /**
     * @brief The name of the field encapsulated by the state object
     */
    std::string name = "";
    /**
     * @brief Whether the GridFunction should be allocated (and owned by the FEState object)
     */
    bool alloc_gf = true;
  };

  /**
   * Main constructor for building a new state object
   * @param[in] mesh The problem mesh (object does not take ownership)
   * @param[in] options The options specified, namely those relating to the order of the problem,
   * the dimension of the FESpace, the type of FEColl, the DOF ordering that should be used,
   * and the name of the field
   */
  FiniteElementState(
      mfem::ParMesh& mesh,
      Options&&      options = {
          .order = 1, .vector_dim = 1, .coll = {}, .ordering = mfem::Ordering::byVDIM, .name = "", .alloc_gf = true});

  /**
   * @brief Minimal constructor for a FiniteElementState given an already-existing field
   * @param[in] mesh The problem mesh (object does not take ownership)
   * @param[in] gf The field for the state to create (object does not take ownership)
   * @param[in] name The name of the field
   */
  FiniteElementState(mfem::ParMesh& mesh, mfem::ParGridFunction& gf, const std::string& name = "");

  /**
   * Returns the MPI communicator for the state
   */
  MPI_Comm comm() const { return detail::retrieve(space_).GetComm(); }

  /**
   * Returns a non-owning reference to the internal grid function
   */
  mfem::ParGridFunction& gridFunc() { return detail::retrieve(gf_); }
  /// \overload
  const mfem::ParGridFunction& gridFunc() const { return detail::retrieve(gf_); }

  /**
   * Returns a GridFunctionCoefficient referencing the internal grid function
   */
  mfem::GridFunctionCoefficient gridFuncCoef() const
  {
    const auto& gf = detail::retrieve(gf_);
    return mfem::GridFunctionCoefficient{&gf, gf.VectorDim()};
  }

  /**
   * Returns a VectorGridFunctionCoefficient referencing the internal grid function
   */
  mfem::VectorGridFunctionCoefficient vectorGridFuncCoef() const
  {
    return mfem::VectorGridFunctionCoefficient{&detail::retrieve(gf_)};
  }

  /**
   * Returns a non-owning reference to the internal mesh object
   */
  mfem::ParMesh& mesh() { return mesh_; }

  /**
   * Returns a non-owning reference to the internal FESpace
   */
  mfem::ParFiniteElementSpace& space() { return detail::retrieve(space_); }
  /// \overload
  const mfem::ParFiniteElementSpace& space() const { return detail::retrieve(space_); }

  /**
   * Returns a non-owning reference to the vector of true DOFs
   */
  mfem::HypreParVector& trueVec() { return true_vec_; }

  /**
   * Returns the name of the FEState (field)
   */
  std::string name() const { return name_; }

  /**
   * Projects a coefficient (vector or scalar) onto the field
   * @param[in] coef The coefficient to project
   */
  void project(const GeneralCoefficient& coef)
  {
    // The generic lambda parameter, auto&&, allows the component type (mfem::Coef or mfem::VecCoef)
    // to be deduced, and the appropriate version of ProjectCoefficient is dispatched.
    visit([this](auto&& concrete_coef) { detail::retrieve(gf_).ProjectCoefficient(*concrete_coef); }, coef);
  }
  /// \overload
  void project(mfem::Coefficient& coef) { detail::retrieve(gf_).ProjectCoefficient(coef); }
  /// \overload
  void project(mfem::VectorCoefficient& coef) { detail::retrieve(gf_).ProjectCoefficient(coef); }

  /**
   * Initialize the true DOF vector by extracting true DOFs from the internal
   * grid function into the internal true DOF vector
   */
  void initializeTrueVec() { detail::retrieve(gf_).GetTrueDofs(true_vec_); }

  /**
   * Set the internal grid function using the true DOF values
   */
  void distributeSharedDofs() { detail::retrieve(gf_).SetFromTrueDofs(true_vec_); }

  /**
   * Utility function for creating a tensor, e.g. mfem::HypreParVector,
   * mfem::ParBilinearForm, etc on the FESpace encapsulated by an FEState object
   * @return An owning pointer to a heap-allocated tensor
   * @pre Tensor must have the constructor Tensor::Tensor(ParFiniteElementSpace*)
   */
  template <typename Tensor>
  std::unique_ptr<Tensor> createOnSpace()
  {
    static_assert(std::is_constructible_v<Tensor, mfem::ParFiniteElementSpace*>,
                  "Tensor must be constructible with a ptr to ParFESpace");
    return std::make_unique<Tensor>(&detail::retrieve(space_));
  }

private:
  /**
   * @brief A reference to the mesh object on which the field is defined
   */
  std::reference_wrapper<mfem::ParMesh> mesh_;
  /**
   * @brief Possibly-owning handle to the FiniteElementCollection, as it is owned
   * by the FiniteElementState in a normal run and by the MFEMSidreDataCollection
   * in a restart run
   * @note Must be const as FESpaces store a const reference to their FEColls
   */
  detail::MaybeOwningPointer<const mfem::FiniteElementCollection> coll_;
  /**
   * @brief Possibly-owning handle to the FiniteElementCollection, as it is owned
   * by the FiniteElementState in a normal run and by the MFEMSidreDataCollection
   * in a restart run
   */
  detail::MaybeOwningPointer<mfem::ParFiniteElementSpace> space_;
  /**
   * @brief Possibly-owning handle to the ParGridFunction, as it is owned
   * by the FiniteElementState in a normal run and by the MFEMSidreDataCollection
   * in a restart run
   */
  detail::MaybeOwningPointer<mfem::ParGridFunction> gf_;
  mfem::HypreParVector                              true_vec_;
  std::string                                       name_ = "";
};

}  // namespace serac
