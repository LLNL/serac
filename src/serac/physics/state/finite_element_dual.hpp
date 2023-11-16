// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file finite_element_dual.hpp
 *
 * @brief This contains a class that represents the dual of a finite element vector space, i.e. the space of residuals
 * and sensitivities.
 */

#pragma once

#include <memory>

#include "mfem.hpp"

#include "serac/physics/state/finite_element_vector.hpp"

namespace serac {

/**
 * @brief Class for encapsulating the dual vector space of a finite element space (i.e. the
 * space of linear forms as applied to a specific basis set)
 */
class FiniteElementDual : public FiniteElementVector {
public:
  using FiniteElementVector::FiniteElementVector;
  using mfem::Vector::Print;

  /**
   * @brief Copy constructor
   *
   * @param[in] rhs The input Dual used for construction
   */
  FiniteElementDual(const FiniteElementDual& rhs) : FiniteElementVector(rhs) {}

  /**
   * @brief Move construct a new Finite Element Dual object
   *
   * @param[in] rhs The input vector used for construction
   */
  FiniteElementDual(FiniteElementDual&& rhs) : FiniteElementVector(std::move(rhs)) {}

  /**
   * @brief Copy assignment
   *
   * @param rhs The right hand side input Dual
   * @return The assigned FiniteElementDual
   */
  FiniteElementDual& operator=(const FiniteElementDual& rhs)
  {
    FiniteElementVector::operator=(rhs);
    return *this;
  }

  /**
   * @brief Move assignment
   *
   * @param rhs The right hand side input Dual
   * @return The assigned FiniteElementDual
   */
  FiniteElementDual& operator=(FiniteElementDual&& rhs)
  {
    std::unique_ptr<mfem::ParLinearForm> lf(std::move(rhs.linear_form_));

    FiniteElementVector::operator=(rhs);

    this->linear_form_ = std::move(lf);
    return *this;
  }

  /**
   * @brief Copy assignment with HypreParVector
   *
   * @param rhs The right hand side input HypreParVector
   * @return The assigned FiniteElementDual
   */
  FiniteElementDual& operator=(const mfem::HypreParVector& rhs)
  {
    FiniteElementVector::operator=(rhs);
    return *this;
  }

  /**
   * @brief Copy assignment with mfem::Vector
   *
   * @param rhs The right hand side input Dual
   * @return The assigned FiniteElementDual
   */
  FiniteElementDual& operator=(const mfem::Vector& rhs)
  {
    FiniteElementVector::operator=(rhs);
    return *this;
  }

  /**
   * @brief Copy assignment with double
   *
   * @param rhs The right hand side input double
   * @return The assigned FiniteElementDual
   */
  FiniteElementDual& operator=(double rhs)
  {
    FiniteElementVector::operator=(rhs);
    return *this;
  }

  /**
   * @brief Fill a user-provided linear form based on the underlying true vector
   *
   * This distributes true vector dofs to the finite element (local) dofs by multiplying the true dofs
   * by the restriction transpose operator.
   *
   * @see <a href="https://mfem.org/pri-dual-vec/">MFEM documentation</a> for details
   *
   * @param linear_form The linear form used to initialize the underlying true vector.
   */
  void fillLinearForm(mfem::ParLinearForm& linear_form) const
  {
    space_->GetRestrictionMatrix()->MultTranspose(*this, linear_form);
  }

  /**
   * @brief Initialize the true vector in the FiniteElementDual based on an input linear form
   *
   * This distributes the linear form dofs to the true vector dofs by multiplying by the
   * prolongation transpose operator.
   *
   * @see <a href="https://mfem.org/pri-dual-vec/">MFEM documentation</a> for details
   *
   * @note The underlying MFEM call should really be const
   *
   * @param linear_form The linear form used to initialize the underlying true vector.
   */
  void setFromLinearForm(const mfem::ParLinearForm& linear_form)
  {
    const_cast<mfem::ParLinearForm&>(linear_form).ParallelAssemble(*this);
  }

  /**
   * @brief Construct a linear form from the finite element dual true vector
   *
   * @return The constructed linear form
   */
  mfem::ParLinearForm& linearForm() const
  {
    if (!linear_form_) {
      linear_form_ = std::make_unique<mfem::ParLinearForm>(space_.get());
    }

    fillLinearForm(*linear_form_);
    return *linear_form_;
  }

protected:
  /**
   * @brief An optional container for a linear form (L-vector) view of the finite element dual.
   *
   * If a user requests it, it is constructed and potentially reused during subsequent calls. It is
   * not updated unless specifically requested via the @a linearForm method.
   */
  mutable std::unique_ptr<mfem::ParLinearForm> linear_form_;
};

}  // namespace serac
