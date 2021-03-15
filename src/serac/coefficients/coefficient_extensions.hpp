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

  namespace detail {
    template <typename T>
    struct index_t {
      using type = std::size_t;
    };
  
    template <typename T>
    struct index_t<mfem::Array<T>> {
      using type = int;
    };

    template <typename T>
    int size(T container) { return static_cast<int> ( container.size() ); }

    template <typename T>
    int size(mfem::Array<T> container) { return container.Size(); }
    
  }
  
  namespace digitize {
    /**
     * @brief takes floating point value and rounds down to the nearest integer
     * @param[in] v floating point value
     */
    
    [[maybe_unused]] static int floor(double v) {
      return static_cast<int>(v);
    }


    /**
     * @brief Thresholds a real function so that it is 2 if greater than 0.
     * @param[in] v floating point value
     */
    [[maybe_unused]] static int threshold(double v) { return v > 0. ? 2 : 1; }

    /**
     * @brief Checks if floating point value is equal to 1, if return 2 otherwise return 1.
     */
    [[maybe_unused]] static int equals1(double v) { return v == 1. ? 2 : 1; }    
  }
  
/**
 * @brief MakeTrueEssList takes in a FESpace, a vector coefficient, and produces a list
 *  of essential boundary conditions
 *
 * @param[in] pfes A finite element space for the constrained grid function
 * @param[in] c A VectorCoefficient that is projected on to the mesh. All
 * d.o.f's are examined and those that are the condition (> 0.) are appended to
 * the vdof list.
 * @return The list of true dofs that should be part of the essential boundary conditions
 */
mfem::Array<int> MakeTrueEssList(mfem::ParFiniteElementSpace& pfes, mfem::VectorCoefficient& c);

/**
 * @brief MakeEssList takes in a FESpace, a vector coefficient, and produces a list
 * of essential boundary conditions
 *
 * @param[in] pfes A finite element space for the constrained grid function
 * @param[in] c A VectorCoefficient that is projected on to the mesh. All
 * d.o.f's are examined and those that are the condition (> 0.) are appended to
 * the vdof list.
 * @return The list of vector dofs that should be
 * part of the essential boundary conditions
 */
mfem::Array<int> MakeEssList(mfem::ParFiniteElementSpace& pfes, mfem::VectorCoefficient& c);

/**
 * @brief This method creates an array of size(local_elems), and assigns
 * attributes based on the coefficient c
 *
 * This method is useful for creating lists of attributes that correspond to
 * elements in the mesh
 *
 * @param[in] m The mesh
 * @param[in] c The coefficient provided that will be evaluated on the mesh
 * @param[in] digitize An optional function that can be
 * called to assign attributes based on the value of c at a given projection
 * point. By default, values of c at a given d.o.f that are > 0. are assigned
 * attribute 2, otherwise attribute 1.
 * @return An array holding the attributes that correspond to each element
 */

  template <typename T>
  T MakeAttributeList(mfem::Mesh& m, mfem::Coefficient& c, std::function<int(double)> digitize = digitize::threshold)
{
  mfem::L2_FECollection    l2_fec(0, m.SpaceDimension());
  mfem::FiniteElementSpace fes(&m, &l2_fec);
  T attr_list(static_cast<typename detail::index_t<T>::type>(fes.GetNE()));

  mfem::GridFunction elem_attr(&fes);
  elem_attr.ProjectCoefficient(c);

  for (auto e = attr_list.begin(); e != attr_list.end(); e++)
    {
      *e = digitize(elem_attr[ e - attr_list.begin() ]);
    }
  
  return attr_list;
}

  
  /**
   * @brief Assign element attributes to mesh
   *
   * @param[in] m the mesh
   * @param[in] attr_list a list of attributes to assign to the given mesh
   */

  template <typename T>
  void AssignMeshElementAttributes(mfem::Mesh &m, T &&list)
   {
    // check to make sure the lists are match the number of elements     
     SLIC_ERROR_IF(detail::size(list) != m.GetNE(), "list size does not match the number of mesh elements");
    
    for (int e = 0; e < m.GetNE(); e++ ) {
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
  // Need to use H1_fec because boundary elements don't exist in L2
  mfem::H1_FECollection    h1_fec(1, m.SpaceDimension());
  mfem::FiniteElementSpace fes(&m, &h1_fec);
  T                        attr_list(fes.GetNBE());
  mfem::Vector             elem_attr(fes.GetNBE());

  for (int e = 0; e < fes.GetNBE(); e++) {
    mfem::Vector dofs(fes.GetBE(e)->GetDof());
    fes.GetBE(e)->Project(c, *fes.GetBdrElementTransformation(e), dofs);
    elem_attr[e] = dofs.Sum() / (dofs.Size() * 1.);
    attr_list[e] = digitize(elem_attr[e]);
  }

  return attr_list;
}

 
  /**
   * @brief Assign bdr element attributes to mesh
   *
   * @param[in] m the mesh
   * @param[in] attr_list a list of attributes to assign to the given mesh
   */

  template <typename T>
  void AssignMeshBdrAttributes(mfem::Mesh &m, T & list)
  {
    // check to make sure the lists are match the number of elements
    SLIC_ERROR_IF(detail::size(list) != m.GetNBE(), "list size does not match the number of mesh elements");
    
    for (int e = 0; e < m.GetNBE(); e++ ) {
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
class AttributeModifierCoefficient : public mfem::Coefficient {
public:
  /**
   * @brief This class temporarily changes the attribute during coefficient
   * evaluation based on a given list.
   *
   * @param[in] attr_list A list of attributes values corresponding to the type
   * of coefficient at each element.
   * @param[in] c The coefficient to "modify" the element attributes
   */
  AttributeModifierCoefficient(const mfem::Array<int>& attr_list, mfem::Coefficient& c)
      : attr_list_(attr_list), coef_(c)
  {
  }

  /**
   * @brief Evaluate the coefficient at a quadrature point
   *
   * @param[in] Tr The element transformation for the evaluation
   * @param[in] ip The integration point for the evaluation
   * @return The value of the coefficient at the quadrature point
   */
  virtual double Eval(mfem::ElementTransformation& Tr, const mfem::IntegrationPoint& ip);

protected:
  /**
   * @brief A list of attributes values corresponding to the type
   * of coefficient at each element.
   */
  const mfem::Array<int>& attr_list_;

  /**
   * @brief The coefficient to "modify" the element attributes
   */
  mfem::Coefficient& coef_;
};

/**
 * @brief Applies various operations to modify a
 * VectorCoefficient
 */
class TransformedVectorCoefficient : public mfem::VectorCoefficient {
public:
  /**
   * @brief Apply a vector function, Func, to v1
   *
   * @param[in] v1 A VectorCoefficient to apply Func to
   * @param[in] func A function that takes in an input vector, and returns the
   * output as the second argument.
   */
  TransformedVectorCoefficient(std::shared_ptr<mfem::VectorCoefficient>                v1,
                               std::function<void(const mfem::Vector&, mfem::Vector&)> func);

  /**
   * @brief Apply a vector function, Func, to v1 and v2
   *
   * @param[in] v1 A VectorCoefficient to apply Func to
   * @param[in] v2 A VectorCoefficient to apply Func to
   * @param[in] func A function that takes in two input vectors, and returns the
   * output as the third argument.
   */
  TransformedVectorCoefficient(std::shared_ptr<mfem::VectorCoefficient> v1, std::shared_ptr<mfem::VectorCoefficient> v2,
                               std::function<void(const mfem::Vector&, const mfem::Vector&, mfem::Vector&)> func);

  /**
   * @brief Evaluate the coefficient at a quadrature point
   *
   * @param[out] V The evaluated coefficient vector at the quadrature point
   * @param[in] T The element transformation for the evaluation
   * @param[in] ip The integration point for the evaluation
   */
  virtual void Eval(mfem::Vector& V, mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip);

private:
  /**
   * @brief The first vector coefficient in the transformation
   */
  std::shared_ptr<mfem::VectorCoefficient> v1_;

  /**
   * @brief The first vector coefficient in the transformation
   */
  std::shared_ptr<mfem::VectorCoefficient> v2_;

  /**
   * @brief The one argument function for a transformed coefficient
   */
  std::function<void(const mfem::Vector&, mfem::Vector&)> mono_function_;

  /**
   * @brief The two argument function for a transformed coefficient
   */
  std::function<void(const mfem::Vector&, const mfem::Vector&, mfem::Vector&)> bi_function_;
};

/**
 * @brief TransformedScalarCoefficient applies various operations to modify a
 * scalar Coefficient
 */
class TransformedScalarCoefficient : public mfem::Coefficient {
public:
  /**
   * @brief Apply a scalar function, Func, to s1
   *
   * @param[in] s1 A Coefficient to apply Func to
   * @param[in] func A function that takes in an input scalar, and returns the
   * output.
   */
  TransformedScalarCoefficient(std::shared_ptr<mfem::Coefficient> s1, std::function<double(const double)> func);

  /**
   * @brief Apply a scalar function, Func, to s1 and s2
   *
   * @param[in] s1 A scalar Coefficient to apply Func to
   * @param[in] s2 A scalar Coefficient to apply Func to
   * @param[in] func A function that takes in two input scalars, and returns the
   * output.
   */
  TransformedScalarCoefficient(std::shared_ptr<mfem::Coefficient> s1, std::shared_ptr<mfem::Coefficient> s2,
                               std::function<double(const double, const double)> func);

  /**
   * @brief Evaluate the coefficient at a quadrature point
   *
   * @param[in] T The element transformation for the evaluation
   * @param[in] ip The integration point for the evaluation
   * @return The value of the coefficient at the quadrature point
   */
  virtual double Eval(mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip);

private:
  /**
   * @brief The first scalar coefficient in the transformation
   */
  std::shared_ptr<mfem::Coefficient> s1_;

  /**
   * @brief The second scalar coefficient in the transformation
   */
  std::shared_ptr<mfem::Coefficient> s2_;

  /**
   * @brief The one argument transformation function
   */
  std::function<double(const double)> mono_function_;

  /**
   * @brief The two argument transformation function
   */
  std::function<double(const double, const double)> bi_function_;
};

}  // namespace serac::mfem_ext
