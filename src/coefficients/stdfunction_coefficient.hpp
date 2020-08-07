// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file stdfunction_coefficient.hpp
 * 
 * @brief MFEM coefficients and helper functions based on std::functions
 */

#ifndef STD_FUNCTION_COEFFICIENT_HPP
#define STD_FUNCTION_COEFFICIENT_HPP

#include <functional>
#include <memory>

#include "mfem.hpp"

namespace serac {

/**
 * @brief StdFunctionCoefficient is an easy way to make an mfem::Coefficient
 *  using a lambda
 *
 * This is a place holder until the coefficient of the same name is merged into
 * mfem.
 */
class StdFunctionCoefficient : public mfem::Coefficient {
 public:

 /**
  * @brief Constructor that takes in an mfem Vector representing the coordinates and produces a double
  * 
  * @param[in] func The scalar-returning std::function to define the coefficient 
  */
  StdFunctionCoefficient(std::function<double(mfem::Vector&)> func);

/**
 * @brief Evalate the coefficient at a quadrature point
 * 
 * @param[in] Tr The element transformation for the evaluation
 * @param[in] ip The integration point for the evaluation 
 * @return The value of the coefficient at the quadrature point
 */
  virtual double Eval(mfem::ElementTransformation& Tr, const mfem::IntegrationPoint& ip);

 private:
 /**
  * @brief The function to evaluate for the coefficient
  */
  std::function<double(mfem::Vector&)> func_;
};

/**
 * @brief StdFunctionVectorCoefficient is an easy way to make an
 * mfem::VectorCoefficient using a lambda
 */
class StdFunctionVectorCoefficient : public mfem::VectorCoefficient {
 public:
  /**
   * @brief StdFunctionVectorCoefficient is an easy way to make an
   * mfem::Coefficient using a lambda
   * 
   * @param[in] dim The dimension of the VectorCoefficient
   * @param[in] func Is a function that matches the following prototype
   * void(mfem::Vector &, mfem::Vector &). The first argument of the function is
   * the position, and the second argument is the output of the function.
   */
  StdFunctionVectorCoefficient(int dim, std::function<void(mfem::Vector&, mfem::Vector&)> func);

/**
 * @brief Evalate the coefficient at a quadrature point
 * 
 * @param[out] V The evaluated coefficient vector at the quadrature point
 * @param[in] T The element transformation for the evaluation
 * @param[in] ip The integration point for the evaluation 
 */
  virtual void Eval(mfem::Vector& V, mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip);

 private:
  /**
  * @brief The function to evaluate for the coefficient
  */
  std::function<void(mfem::Vector&, mfem::Vector&)> func_;
};

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
mfem::Array<int> makeTrueEssList(mfem::ParFiniteElementSpace& pfes, mfem::VectorCoefficient& c);

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
mfem::Array<int> makeEssList(mfem::ParFiniteElementSpace& pfes, mfem::VectorCoefficient& c);

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
mfem::Array<int> makeAttributeList(
    mfem::Mesh& m, mfem::Coefficient& c, std::function<int(double)> = [](double v) { return v > 0. ? 2 : 1; });

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
mfem::Array<int> makeBdrAttributeList(
    mfem::Mesh& m, mfem::Coefficient& c, std::function<int(double)> = [](double v) { return v == 1. ? 2 : 1; });

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
 * @brief Evalate the coefficient at a quadrature point
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
  mfem::Coefficient&      coef_;
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
  TransformedVectorCoefficient(std::shared_ptr<mfem::VectorCoefficient>          v1,
                               std::function<void(mfem::Vector&, mfem::Vector&)> func);

  /**
   * @brief Apply a vector function, Func, to v1 and v2
   * 
   * @param[in] v1 A VectorCoefficient to apply Func to
   * @param[in] v2 A VectorCoefficient to apply Func to
   * @param[in] func A function that takes in two input vectors, and returns the
   * output as the third argument.
   */
  TransformedVectorCoefficient(std::shared_ptr<mfem::VectorCoefficient> v1, std::shared_ptr<mfem::VectorCoefficient> v2,
                               std::function<void(mfem::Vector&, mfem::Vector&, mfem::Vector&)> func);
  
/**
 * @brief Evalate the coefficient at a quadrature point
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
  std::function<void(mfem::Vector&, mfem::Vector&)>                mono_function_;

  /**
  * @brief The two argument function for a transformed coefficient
  */
  std::function<void(mfem::Vector&, mfem::Vector&, mfem::Vector&)> bi_function_;
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
  * @brief Evalate the coefficient at a quadrature point
  * 
  * @param[in] Tr The element transformation for the evaluation
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
  std::function<double(const double)>               mono_function_;

  /**
   * @brief The two argument transformation function
   */
  std::function<double(const double, const double)> bi_function_;
};

}  // namespace serac

#endif
