// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file coefficient_extensions.hpp
 *
 * @brief Extensions of MFEM's coefficient interface and helper functions
 */

#pragma once

#include <functional>
#include <memory>
#include <variant>
#include <vector>
#include "mfem.hpp"

#include "serac/numerics/expr_template_ops.hpp"

/**
 * @brief Functionality that extends current MFEM capabilities
 *
 */
namespace serac::mfem_ext {

/// @brief Internal implementation details namespace
namespace detail {

// methods for determining index type

/// @brief Provides the member typedef type for the index of type T
template <typename T>
struct index_t {
  /// @brief Member typedef type
  using type = std::size_t;
};

/// @brief Provides the member typedef type for the index of mfem::Arrays
template <typename T>
struct index_t<mfem::Array<T>> {
  /// @brief Member typedef type
  using type = int;
};

// Methods for determining the size of a container

/// @brief Returns the size of container T
template <typename T>
auto size(T&& container)
{
  return container.size();
}

/// @brief Returns the size of mfem::Array
template <typename T>
std::size_t size(const mfem::Array<T>& container)
{
  return container.Size();
}

// Methods for determining type of coefficient evaluations

/// @brief Returns return type of POD-type T
template <typename T, typename = void>
struct eval_result_t {
  /// @brief POD-type
  using type = T;
};

/// @brief Returns return type for mfem::Coefficient
template <typename T>
struct eval_result_t<T, std::enable_if_t<std::is_base_of_v<mfem::Coefficient, T>>> {
  /// @brief mfem::Coefficient return type
  using type = double;
};

/// @brief Returns return type for mfem::VectorCefficient
template <typename T>
struct eval_result_t<T, std::enable_if_t<std::is_base_of_v<mfem::VectorCoefficient, T>>> {
  /// @brief mfem::VectorCoefficient return type
  using type = mfem::Vector;
};

// Methods for evaluating coefficient stuff with the same prototype

/// @brief Returns d unevaluated
double eval(double& d, mfem::ElementTransformation&, const mfem::IntegrationPoint&) { return d; }

/// @brief evaluates an mfem::Coefficient
double eval(mfem::Coefficient& c, mfem::ElementTransformation& Tr, const mfem::IntegrationPoint& ip)
{
  return c.Eval(Tr, ip);
}

/// @brief evaluates a mfem::VectorCoefficient
mfem::Vector eval(mfem::VectorCoefficient& v, mfem::ElementTransformation& Tr, const mfem::IntegrationPoint& ip)
{
  mfem::Vector temp(v.GetVDim());
  v.Eval(temp, Tr, ip);
  return temp;
}

template <typename T, typename = void>
struct is_iterable : std::false_type {
};

template <typename T>
struct is_iterable<T, std::void_t<decltype(std::declval<T>().begin()), decltype(std::declval<T>().end())>>
    : std::true_type {
};

template <typename T, typename = void>
struct has_brackets : std::false_type {
};

template <typename T>
struct has_brackets<T, std::void_t<decltype(std::declval<T>()[0])>> : std::true_type {
};

}  // namespace detail

/**
 * @brief This namespace provides a set of commonly used "digitizing" functions for convenience.
 *
 * The goal of a "digitizing" function is to convert a floating-point number to an integer to be used as an mfem
 * attribute. Mfem attributes must be integers > 0 (1,2,... etc).
 */
namespace digitize {
/**
 * @brief takes floating point value and rounds down to the nearest integer
 * @param[in] v floating point value
 */

[[maybe_unused]] int floor(double v) { return static_cast<int>(std::floor(v)); }

/**
 * @brief Returns 2 if v > 0 and 1 otherwise.
 * @param[in] v floating point value
 */
[[maybe_unused]] int greater_than_zero(double v) { return v > 0. ? 2 : 1; }

/**
 * @brief Checks if floating point value is equal to 1, if return 2 otherwise return 1.
 * @param[in] v floating point value
 */
[[maybe_unused]] int equals1(double v) { return v == 1. ? 2 : 1; }
}  // namespace digitize

/**
 * @brief MakeEssList takes in a FESpace, a vector coefficient, and produces a list
 * of essential boundary conditions
 *
 * @tparam T A mfem::Coefficient or a mfem::VectorCoefficient
 * @param[in] pfes A finite element space for the constrained grid function
 * @param[in] c A coefficient that is projected on to the mesh. All
 * d.o.f's are examined and those that are the condition (> 0.) are appended to
 * the vdof list.
 * @return The list of vector dofs that should be
 * part of the essential boundary conditions
 */
template <typename T, typename SFINAE = std::enable_if_t<std::is_base_of_v<mfem::Coefficient, T> ||
                                                         std::is_base_of_v<mfem::VectorCoefficient, T>>>
mfem::Array<int> MakeEssList(mfem::ParFiniteElementSpace& pfes, T& c)
{
  mfem::Array<int> ess_vdof_list;

  mfem::ParGridFunction v_attr(&pfes);
  v_attr.ProjectCoefficient(c);

  for (int vdof = 0; vdof < pfes.GetVSize(); ++vdof) {
    if (v_attr[vdof] > 0.) {
      ess_vdof_list.Append(vdof);
    }
  }

  return ess_vdof_list;
}

/**
 * @brief MakeTrueEssList takes in a FESpace, a vector coefficient, and produces a list
 *  of essential boundary conditions
 *
 * @tparam T A mfem::Coefficient or a mfem::VectorCoefficient
 * @param[in] pfes A finite element space for the constrained grid function
 * @param[in] c A VectorCoefficient that is projected on to the mesh. All
 * d.o.f's are examined and those that are the condition (> 0.) are appended to
 * the vdof list.
 * @return The list of true dofs that should be part of the essential boundary conditions
 */
template <typename T, typename SFINAE = std::enable_if_t<std::is_base_of_v<mfem::Coefficient, T> ||
                                                         std::is_base_of_v<mfem::VectorCoefficient, T>>>
mfem::Array<int> MakeTrueEssList(mfem::ParFiniteElementSpace& pfes, T& c)
{
  mfem::Array<int> ess_tdof_list;

  mfem::Array<int> ess_vdof_list = MakeEssList(pfes, c);

  for (int i = 0; i < ess_vdof_list.Size(); ++i) {
    int tdof = pfes.GetLocalTDofNumber(ess_vdof_list[i]);
    if (tdof >= 0) {
      ess_tdof_list.Append(tdof);
    }
  }

  return ess_tdof_list;
}

/**
 * @brief This method creates an array of size(local_elems), and assigns
 * attributes based on the coefficient c
 *
 * This method is useful for creating lists of attributes that correspond to
 * elements in the mesh
 *
 * @pre The template type, T, should have a constructor that takes in the size, and
 * should have begin() and end() methods.
 *
 * @pre T must be constructible with an instance of detail::index_t
 *
 * @tparam T Return type is either a suitable std collection or mfem::Array
 * @param[in] m The mesh
 * @param[in] c The coefficient provided that will be evaluated on the mesh
 * @param[in] digitize A function mapping coefficient values onto integer attribute values.
 * By default, values of c at a given point that are greater than zero ( > 0) are assigned
 * attribute 2, otherwise attribute 1.
 * @return An array holding the attributes that correspond to each element
 */

template <typename T>
T MakeAttributeList(mfem::Mesh& m, mfem::Coefficient& c,
                    std::function<int(double)> digitize = digitize::greater_than_zero)
{
  static_assert(std::is_constructible<T, typename detail::index_t<T>::type>::value,
                "T does not have a constructor of type detail::index_t<T>");
  static_assert(detail::is_iterable<T>::value, "T does not implement begin() and end()");

  mfem::L2_FECollection    l2_fec(0, m.SpaceDimension());
  mfem::FiniteElementSpace fes(&m, &l2_fec);
  T                        attr_list(static_cast<typename detail::index_t<T>::type>(fes.GetNE()));

  mfem::GridFunction elem_attr(&fes);
  elem_attr.ProjectCoefficient(c);

  for (auto e = attr_list.begin(); e != attr_list.end(); e++) {
    *e = digitize(elem_attr[e - attr_list.begin()]);
  }

  return attr_list;
}

/**
 * @brief Assign element attributes to mesh
 *
 * @tparam T a container that implements operator[] and size() or mfem::Array
 * @param[in] m the mesh
 * @param[in] list a list of attributes to assign to the given mesh
 */

template <typename T>
void AssignMeshElementAttributes(mfem::Mesh& m, T&& list)
{
  static_assert(detail::has_brackets<T>::value, "T does not contain operator[]");

  // check to make sure the lists are match the number of elements
  SLIC_ERROR_IF(detail::size(list) != static_cast<std::size_t>(m.GetNE()),
                "list size does not match the number of mesh elements");

  for (int e = 0; e < m.GetNE(); e++) {
    m.GetElement(e)->SetAttribute(list[static_cast<typename detail::index_t<T>::type>(e)]);
  }
  m.SetAttributes();
}

/**
 * @brief This method creates an array of size(local_bdr_elems), and assigns
 * attributes based on the coefficient c
 *
 * This method is useful for creating lists of attributes that correspond to bdr
 * elements in the mesh
 *
 * @pre The template type, T, should have a constructor that takes in the size, and
 * should have begin() and end() methods.
 *
 * @tparam T Return type is either a suitable std collection or mfem::Array
 * @param[in] m The mesh
 * @param[in] c The coefficient provided that will be evaluated on the mesh
 * @param[in] digitize An optional function that can be
 * called to assign attributes based on the value of c at a given projection
 * point. By default, values of c at a given d.o.f that are ==1. are assigned
 * attribute 2, otherwise attribute 1. This means that only if all the d.o.f's
 * of an bdr_element are "tagged" 1, will this bdr element be assigned
 * attribute 2.
 * @return An array holding the attributes that correspond to each element
 */
template <typename T>
T MakeBdrAttributeList(mfem::Mesh& m, mfem::Coefficient& c, std::function<int(double)> digitize = digitize::equals1)
{
  static_assert(std::is_constructible<T, typename detail::index_t<T>::type>::value,
                "T does not have a constructor of type detail::index_t<T>");

  static_assert(detail::is_iterable<T>::value, "T does not implement begin() and end()");

  // Need to use H1_fec because boundary elements don't exist in L2
  mfem::H1_FECollection    h1_fec(1, m.SpaceDimension());
  mfem::FiniteElementSpace fes(&m, &h1_fec);
  T                        attr_list(static_cast<typename detail::index_t<T>::type>(fes.GetNBE()));
  mfem::Vector             elem_attr(fes.GetNBE());

  for (auto e = attr_list.begin(); e != attr_list.end(); e++) {
    int          index = static_cast<int>(e - attr_list.begin());
    mfem::Vector dofs(fes.GetBE(index)->GetDof());
    fes.GetBE(index)->Project(c, *fes.GetBdrElementTransformation(index), dofs);
    elem_attr[index] = dofs.Sum() / (dofs.Size() * 1.);
    *e               = digitize(elem_attr[index]);
  }

  return attr_list;
}

/**
 * @brief Assign bdr element attributes to mesh
 *
 * @pre T must implement operator[]
 *
 * @param[in] m the mesh
 * @param[in] list a list of attributes to assign to the given mesh
 */

template <typename T>
void AssignMeshBdrAttributes(mfem::Mesh& m, T&& list)
{
  static_assert(detail::has_brackets<T>::value, "T does not contain operator[]");

  // check to make sure the lists are match the number of elements
  SLIC_ERROR_IF(detail::size(list) != static_cast<std::size_t>(m.GetNBE()),
                "list size does not match the number of mesh elements");

  for (int e = 0; e < m.GetNBE(); e++) {
    m.GetBdrElement(e)->SetAttribute(list[static_cast<typename detail::index_t<T>::type>(e)]);
  }
  m.SetAttributes();
}

/**
 * @brief AttributemodifierCoefficient class
 *
 * This class temporarily changes the attribute to a given attribute list during
 * evaluation
 */
template <typename T>
class AttributeModifierCoefficient : public mfem::Coefficient {
public:
  /**
   * @brief This class temporarily changes the attribute during coefficient
   * evaluation based on a given list.
   *
   * @pre T must implement operator[]
   *
   * @tparam T A suitable list std collection or mfem::Array
   * @param[in] attr_list A list of attributes values corresponding to the type
   * of coefficient at each element.
   * @param[in] c The coefficient to "modify" the element attributes
   */
  AttributeModifierCoefficient(const T& attr_list, mfem::Coefficient& c) : attr_list_(attr_list), coef_(c)
  {
    static_assert(detail::has_brackets<T>::value, "T does not contain operator[]");
  }

  /**
   * @brief Evaluate the coefficient at a quadrature point
   *
   * @param[in] Tr The element transformation for the evaluation
   * @param[in] ip The integration point for the evaluation
   * @return The value of the coefficient at the quadrature point
   */
  double Eval(mfem::ElementTransformation& Tr, const mfem::IntegrationPoint& ip) override
  {
    // Store old attribute and change to new attribute
    const int attr = Tr.Attribute;
    Tr.Attribute   = attr_list_[static_cast<typename detail::index_t<T>::type>(Tr.ElementNo)];

    // Evaluate with new attribute
    double result = coef_.Eval(Tr, ip);

    // Set back to original attribute (maybe it's not necessary?.. just to be
    // safe)
    Tr.Attribute = attr;

    return result;
  }

protected:
  /**
   * @brief A list of attributes values corresponding to the type
   * of coefficient at each element.
   */
  const T& attr_list_;

  /**
   * @brief The coefficient to "modify" the element attributes
   */
  mfem::Coefficient& coef_;
};

/**
 * @brief AttributemodifierCoefficient class
 *
 * This class temporarily changes the attribute to a given attribute list during
 * evaluation
 */
template <typename T>
class AttributeModifierVectorCoefficient : public mfem::VectorCoefficient {
public:
  /**
   * @brief This class temporarily changes the attribute during coefficient
   * evaluation based on a given list.
   *
   * @pre T must implement operator[]
   *
   * @tparam T A suitable list std collection or mfem::Array
   * @param[in] dim Vector dimensions
   * @param[in] attr_list A list of attributes values corresponding to the type
   * of coefficient at each element.
   * @param[in] c The coefficient to "modify" the element attributes
   */
  AttributeModifierVectorCoefficient(int dim, const T& attr_list, mfem::VectorCoefficient& c)
      : attr_list_(attr_list), coef_(c), mfem::VectorCoefficient(dim)
  {
    static_assert(detail::has_brackets<T>::value, "T does not contain operator[]");
  }

  /**
   * @brief Evaluate the coefficient at a quadrature point
   *
   * @param[in] v  The evaluated coefficient vector at the quadrature point
   * @param[in] Tr The element transformation for the evaluation
   * @param[in] ip The integration point for the evaluation
   * @return The value of the coefficient at the quadrature point
   */
  void Eval(mfem::Vector& v, mfem::ElementTransformation& Tr, const mfem::IntegrationPoint& ip) override
  {
    // Store old attribute and change to new attribute
    const int attr = Tr.Attribute;
    Tr.Attribute   = attr_list_[static_cast<typename detail::index_t<T>::type>(Tr.ElementNo)];

    // Evaluate with new attribute
    coef_.Eval(v, Tr, ip);

    // Set back to original attribute (maybe it's not necessary?.. just to be
    // safe)
    Tr.Attribute = attr;
  }

protected:
  /**
   * @brief A list of attributes values corresponding to the type
   * of coefficient at each element.
   */
  const T& attr_list_;

  /**
   * @brief The coefficient to "modify" the element attributes
   */
  mfem::VectorCoefficient& coef_;
};

/**
 * @brief TransformedVectorCoefficient applies various operations to modify a list of arguments
 */
template <typename... Types>
class TransformedVectorCoefficient : public mfem::VectorCoefficient {
public:
  /**
   * @brief Apply a vector function, Func, to args_1.... args_n
   *
   * @pre The arguments must correspond to the function signature of the supplied function:
   * e.g. (mfem::Coefficient -> double), (mfem::VectorCoefficient -> mfem::Vector), or POD-numeric types which return
   * POD-numeric types
   *
   *
   * @param[in] dim d.o.f of this mfem::Vectorcoefficient
   * @param[in] func A function to apply to the evaluations of all args and return mfem::Vector
   * @param[in] args A list of mfem::Coefficients, mfem::VectorCoefficients, or numbers
   */
  TransformedVectorCoefficient(int                                                                          dim,
                               std::function<mfem::Vector(typename detail::eval_result_t<Types>::type&...)> func,
                               Types&... args)
      : mfem::VectorCoefficient(dim), references_(std::make_tuple(std::ref(args)...)), function_(func)
  {
    static_assert(std::is_invocable_v<mfem::Vector(typename detail::eval_result_t<Types>::type & ...),
                                      typename detail::eval_result_t<Types>::type&...>);
  }

  /**
   * @brief Evaluate the coefficient at a quadrature point
   *
   * @param[out] V The evaluated coefficient vector at the quadrature point
   * @param[in] Tr The element transformation for the evaluation
   * @param[in] ip The integration point for the evaluation
   */
  void Eval(mfem::Vector& V, mfem::ElementTransformation& Tr, const mfem::IntegrationPoint& ip) override
  {
    // Evaluate all the references types
    V.SetSize(GetVDim());
    auto results =
        std::apply([&](auto&&... args) { return std::make_tuple(detail::eval(args.get(), Tr, ip)...); }, references_);
    V = std::apply(function_, results);
  }

private:
  /**
   * @brief tuple of references to input arguments
   */
  std::tuple<std::reference_wrapper<Types>...> references_;

  /**
   * @brief function to return a vector on evaluated arguments
   */
  std::function<mfem::Vector(typename detail::eval_result_t<Types>::type&...)> function_;
};

/**
 * @brief TransformedScalarCoefficient is a coefficient that applies various operations to modify a list of arguments
 */
template <typename... Types>
class TransformedScalarCoefficient : public mfem::Coefficient {
public:
  /**
   * @brief Apply a scalar function, Func, to args_1.... args_n
   *
   * @pre The arguments must correspond to the function signature of the supplied function:
   * e.g. (mfem::Coefficient -> double), (mfem::VectorCoefficient -> mfem::Vector), or POD-numeric types which return
   * POD-numeric types
   *
   *
   * @param[in] func A function to apply to the evaluations of all args and return double
   * @param[in] args A list of mfem::Coefficients, mfem::VectorCoefficients, or numbers
   */
  TransformedScalarCoefficient(std::function<double(typename detail::eval_result_t<Types>::type&...)> func,
                               Types&... args)
      : mfem::Coefficient(), references_(std::make_tuple(std::ref(args)...)), function_(func)
  {
    static_assert(std::is_invocable_v<double(typename detail::eval_result_t<Types>::type & ...),
                                      typename detail::eval_result_t<Types>::type&...>);
  }

  /**
   * @brief Evaluate the coefficient at a quadrature point
   *
   * @param[in] Tr The element transformation for the evaluation
   * @param[in] ip The integration point for the evaluation
   * @return The value of the coefficient at the quadrature point
   */
  double Eval(mfem::ElementTransformation& Tr, const mfem::IntegrationPoint& ip) override
  {
    auto results =
        std::apply([&](auto&&... args) { return std::make_tuple(detail::eval(args.get(), Tr, ip)...); }, references_);
    return std::apply(function_, results);
  }

private:
  /**
   * @brief tuple of references to input arguments
   */
  std::tuple<std::reference_wrapper<Types>...> references_;

  /**
   * @brief function to return a vector on evaluated arguments
   */
  std::function<double(typename detail::eval_result_t<Types>::type&...)> function_;
};

/**
 * @brief SurfaceElementAttrCoefficient evaluates an element attribute-based coefficient on the surface
 */

class SurfaceElementAttrCoefficient : public mfem::Coefficient {
public:
  /**
   * @brief Evaluates an element attribute-based coefficient on the a boundary element
   *
   * @param[in] mesh The mesh we want to evaluate the element attribute-based coefficient with
   * @param[in] c The element attribute-based coefficient to evaluate on the boundary
   */
  SurfaceElementAttrCoefficient(mfem::ParMesh& mesh, mfem::Coefficient& c) : coef_(c), pmesh_(mesh) {}

  /**
   * @brief Evaluates an element attribute-based coefficient on the a boundary element
   *
   * @param[in] Tr The local surface FE transformation
   * @param[in] ip The current surface element integration point
   * @return The value of the element attribute-based coefficient evaluated on the boundary
   */
  double Eval(mfem::ElementTransformation& Tr, const mfem::IntegrationPoint& ip) override
  {
    // Find attached element
    int bdr_el = Tr.ElementNo;
    int el, face_info;

    pmesh_.GetBdrElementAdjacentElement(bdr_el, el, face_info);

    return coef_.Eval(*pmesh_.GetElementTransformation(el), ip);
  }

protected:
  /**
   * @brief Underlying element attribute-based coefficient
   */
  mfem::Coefficient& coef_;

  /**
   * @brief mfem::ParMesh containing attributes
   */
  mfem::ParMesh& pmesh_;
};

/**
 * @brief SurfaceVectorElementAttrCoefficient evaluates an element attribute-based VectorCoefficient on the surface
 */

class SurfaceVectorElementAttrCoefficient : public mfem::VectorCoefficient {
public:
  /**
   * @brief Evaluates an element attribute-based coefficient on the a boundary element
   *
   * @param[in] mesh The mesh we want to evaluate the element attribute-based coefficient with
   * @param[in] c The element attribute-based VectorCoefficient to evaluate on the boundary
   */
  SurfaceVectorElementAttrCoefficient(mfem::ParMesh& mesh, mfem::VectorCoefficient& c)
      : mfem::VectorCoefficient(c.GetVDim()), coef_(c), pmesh_(mesh)
  {
  }

  /**
   * @brief Evaluates an element attribute-based coefficient on the a boundary element
   *
   * @param[inout] V The vector-value of the element attribute-based coefficient evaluated on the boundary
   * @param[in] Tr The local surface FE transformation
   * @param[in] ip The current surface element integration point
   */
  void Eval(mfem::Vector& V, mfem::ElementTransformation& Tr, const mfem::IntegrationPoint& ip) override
  {
    // Find attached element
    int bdr_el = Tr.ElementNo;
    int el, face_info;

    pmesh_.GetBdrElementAdjacentElement(bdr_el, el, face_info);

    V.SetSize(coef_.GetVDim());

    coef_.Eval(V, *pmesh_.GetElementTransformation(el), ip);
  }

protected:
  /**
   * @brief Underlying element attribute-based mfem::VectorCoefficient
   */
  mfem::VectorCoefficient& coef_;

  /**
   * @brief mfem::ParMesh containing attributes
   */
  mfem::ParMesh& pmesh_;
};

}  // namespace serac::mfem_ext
