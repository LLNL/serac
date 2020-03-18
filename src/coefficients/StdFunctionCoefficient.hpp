// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
// # Author: Jonathan Wong @ LLNL.

/**
    @file StdFunctionCoefficient.hpp
 */

#ifndef STD_FUNCTION_COEFFICIENT_HPP
#define STD_FUNCTION_COEFFICIENT_HPP
#include <functional>
#include "mfem.hpp"

using namespace mfem;

/**
   \brief StdFunctionCoefficient is an easy way to make an mfem::Coefficient using a lambda

   This is a place holder until the coefficient of the same name is merged into mfem.

*/
class StdFunctionCoefficient : public Coefficient
{
public:
  /// Constructor that takes in an mfem Vector representing the coordinates and produces a double
  StdFunctionCoefficient(std::function<double(mfem::Vector &)> func);
  virtual ~StdFunctionCoefficient() {}

  virtual double Eval (ElementTransformation &Tr, const IntegrationPoint &ip);
private:
  std::function<double(mfem::Vector &)> func_;
};

/**
   \brief StdFunctionVectorCoefficient is an easy way to make an mfem::Coefficient using a lambda

*/
class StdFunctionVectorCoefficient : public VectorCoefficient
{
public:

  /**
     \brief StdFunctionVectorCoefficient is an easy way to make an mfem::Coefficient using a lambda

     \param[in] dim The dimension of the VectorCoefficient
     \param[in] func Is a function that matches the following prototype void(mfem::Vector &, mfem::Vector &). The first argument of the function is the position, and the second argument is the output of the function.
  */
  StdFunctionVectorCoefficient(int dim,
                               std::function<void(mfem::Vector &, mfem::Vector &)> func);

  virtual void Eval (Vector &V, ElementTransformation &T, const IntegrationPoint &ip);
private:
  std::function<void(mfem::Vector &,mfem::Vector &)> func_;
};

/**
   \brief MakeEssList takes in a mesh, a vector coefficient, and produces a list of essential boundary conditions

   \param[in] m Mesh
   \param[in] c A VectorCoefficient that is porjected on to the mesh. All d.o.f's are examined and those that are the condition (> 0.) are appended toe the vdof list.
   \param[out] ess_vdof_list The list of vdofs that should be part of the essential boundary conditions
*/
void MakeEssList(Mesh &m, VectorCoefficient &c, Array<int> & ess_vdof_list);

/**
   \brief This method creates an array of size(local_elems), and assigns attributes based on the coefficient c

   This method is useful for creating lists of attributes that correspond to elements in the mesh

   \param[in] m The mesh
   \param[inout] attr_list Should be an array that will hold the attributes that correspond to each element
   \param[in] c The coefficient provided that will be evaluated on the mesh
   \param[in] digitize An optional function that can be called to assign attributes based on the value of c at a given projection point. By default, values of c at a given d.o.f that are > 0. are assigned attribute 2, otherwise attribute 1.

*/
void MakeAttributeList(Mesh &m, Array<int> &attr_list,
                       Coefficient &c,
                       std::function<int(double)> = [](double v)
{
  return v > 0. ? 2 : 1;
});

/**
   \brief This method creates an array of size(local_bdr_elems), and assigns attributes based on the coefficient c

   This method is useful for creating lists of attributes that correspond to bdr elements in the mesh

   \param[in] m The mesh
   \param[inout] attr_list Should be an array that will hold the attributes that correspond to each element
   \param[in] c The coefficient provided that will be evaluated on the mesh
   \param[in] digitize An optional function that can be called to assign attributes based on the value of c at a given projection point. By default, values of c at a given d.o.f that are ==1. are assigned attribute 2, otherwise attribute 1. This means that only if all the d.o.f's of an bdr_element are "tagged" 1, will this bdr element be assigned attribute 2.

*/
void MakeBdrAttributeList(Mesh &m, Array<int> &attr_list,
                          Coefficient &c,
                          std::function<int(double)> = [](double v)
{
  return v == 1. ? 2 : 1;
});

/**
   \brief AttributemodifierCoefficient class

   This class temporarily changes the attribute to a given attribute list during evaluation
*/
class AttributeModifierCoefficient : public Coefficient
{
public:
  /**
     \brief This class temporarily changes the attribute during coefficient evaluation based on a given list.

     \param[in] attr_list A list of attributes values corresponding to the type of coefficient at each element.
     \param[in] c The coefficient to "modify" the element attributes
  */
  AttributeModifierCoefficient(const Array<int> &attr_list, Coefficient &c) :
    attr_list_(attr_list), C_(&c) {}

  virtual double Eval (ElementTransformation &Tr, const IntegrationPoint &ip);

protected:
  const Array<int> & attr_list_;
  Coefficient *C_;
};

/**
   TransformedVectorCoefficient applies various operations to modify a VectorCoefficient
*/
class TransformedVectorCoefficient : public VectorCoefficient
{
public:

  /**
     \brief Apply a vector function, Func, to v1


     \param[in] v1 A VectorCoefficient to apply Func to
     \param[in] func A function that takes in an input vector, and returns the output as the second argument.
  */
  TransformedVectorCoefficient(VectorCoefficient *v1, std::function <void (Vector &, Vector &)> func);

  /**
     \brief Apply a vector function, Func, to v1 and v2


     \param[in] v1 A VectorCoefficient to apply Func to
     \param[in] v2 A VectorCoefficient to apply Func to
     \param[in] func A function that takes in two input vectors, and returns the output as the third argument.
  */

  TransformedVectorCoefficient(VectorCoefficient *v1, VectorCoefficient *v2,
                               std::function <void (Vector &, Vector &, Vector &)>  func);
  virtual void Eval (Vector & V, ElementTransformation & T, const IntegrationPoint & ip );

private:
  VectorCoefficient *v1_;
  VectorCoefficient *v2_;

  std::function <void (Vector &, Vector &)> mono_function_;
  std::function <void (Vector &, Vector &, Vector &)> bi_function_;
};



#endif
