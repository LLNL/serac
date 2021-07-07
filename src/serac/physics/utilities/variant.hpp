// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file variant.hpp
 *
 * @brief This file contains the declaration of a two-element variant type
 */

#pragma once

#include <type_traits>

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
}  // namespace serac
