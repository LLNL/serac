// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "mfem.hpp"

/** Incremental hyperelastic integrator for any given HyperelasticModel.

    Represents @f$ \int W(Jpt) dx @f$ over a target zone, where W is the
    @a model's strain energy density function, and Jpt is the Jacobian of the
    target->physical coordinates transformation. The target configuration is
    given by the current mesh at the time of the evaluation of the integrator.
*/
class IncrementalHyperelasticIntegrator : public mfem::NonlinearFormIntegrator {
 private:
  mfem::HyperelasticModel *model;

  //   Jrt: the Jacobian of the target-to-reference-element transformation.
  //   Jpr: the Jacobian of the reference-to-physical-element transformation.
  //   Jpt: the Jacobian of the target-to-physical-element transformation.
  //     P: represents dW_d(Jtp) (dim x dim).
  //   DSh: gradients of reference shape functions (dof x dim).
  //    DS: gradients of the shape functions in the target (stress-free)
  //        configuration (dof x dim).
  // PMatI: coordinates of the deformed configuration (dof x dim).
  // PMatO: reshaped view into the local element contribution to the operator
  //        output - the result of AssembleElementVector() (dof x dim).
  mfem::DenseMatrix DSh, DS, Jrt, Jpr, Jpt, P, PMatI, PMatO;

 public:
  /** @param[in] m  HyperelasticModel that will be integrated. */
  IncrementalHyperelasticIntegrator(mfem::HyperelasticModel *m) : model(m) {}

  /** @brief Computes the integral of W(Jacobian(Trt)) over a target zone
      @param[in] el     Type of FiniteElement.
      @param[in] Ttr    Represents ref->target coordinates transformation.
      @param[in] elfun  Physical coordinates of the zone. */
  virtual double GetElementEnergy(const mfem::FiniteElement &  el,
                                  mfem::ElementTransformation &Ttr,
                                  const mfem::Vector &         elfun);

  virtual void AssembleElementVector(const mfem::FiniteElement &  el,
                                     mfem::ElementTransformation &Ttr,
                                     const mfem::Vector &         elfun,
                                     mfem::Vector &               elvect);

  virtual void AssembleElementGrad(const mfem::FiniteElement &  el,
                                   mfem::ElementTransformation &Ttr,
                                   const mfem::Vector &         elfun,
                                   mfem::DenseMatrix &          elmat);
};
