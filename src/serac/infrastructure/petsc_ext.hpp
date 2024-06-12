// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
#pragma once

#include "mfem.hpp"
#include "mfem/linalg/petsc.hpp"

#ifdef MFEM_USE_PETSC

#include "petsc.h"
#include "petscpc.h"

namespace serac::mfem_ext {
/**
 * @brief Wrapper around mfem::PetscLinearSolver supporting the mfem::IterativeSolver interface
 */
class PetscKSPSolver : public mfem::IterativeSolver, public mfem::PetscLinearSolver {
private:
  /// @brief The MPI communicator used by the vectors and matrices in the solve
  MPI_Comm comm_;
  /// @brief Flag determining whether the mfem::Operator is wrapped or converted
  bool wrap_ = false;
  /// @brief Cached converted operator
  mfem::PetscParMatrix* pA_ = nullptr;

public:
  /**
   * @brief Construct a wrapper for using a PETSc linear solver
   *
   * @param comm The MPI communicator used by the vectors and matrices in the solve
   * @param ksp_type Type of PETSc linear solver to construct, e.g. KSPCG or KSPGMRES
   * @param prefix Prefix for all PETSc command line options options
   * @param wrap Flag determining whether the mfem::Operator is wrapped or converted, defaults to false
   * @param iter_mode Flag controlling whether the second argument to Mult() is treated as an initial guess
   *
   * @note If @a wrap is true, then the MatMult ops of HypreParMatrix are wrapped.
   *       No preconditioner can be automatically constructed from PETSc.
   *       If @a wrap is false, the HypreParMatrix is converted into a the AIJ
   *       PETSc format, which is suitable for most preconditioning methods.
   */
  PetscKSPSolver(MPI_Comm comm, KSPType ksp_type = KSPCG, const std::string& prefix = std::string(), bool wrap = false,
                 bool iter_mode = false);

  /**
   * @brief Constructs a solver using a HypreParMatrix.
   *
   * @param A The mfem::PetscParMatrix defining the linear system
   * @param ksp_type Type of PETSc linear solver to construct, e.g. KSPCG or KSPGMRES
   * @param prefix Prefix for all PETSc command line options options
   * @param iter_mode Flag controlling whether the second argument to Mult() is treated as an initial guess
   */
  PetscKSPSolver(const mfem::PetscParMatrix& A, KSPType ksp_type = KSPCG, const std::string& prefix = std::string(),
                 bool iter_mode = false);
  /**
   * @brief Constructs a solver using a HypreParMatrix.
   *
   * @param A The mfem::HypreParMatrix to use to construct the iterative linear solver
   * @param ksp_type Type of PETSc linear solver to construct, e.g. KSPCG or KSPGMRES
   * @param prefix Prefix for all PETSc command line options options
   * @param wrap Flag determining whether the mfem::Operator is wrapped or converted, defaults to false
   * @param iter_mode Flag controlling whether the second argument to Mult() is treated as an initial guess
   *
   * @note If @a wrap is true, then the MatMult ops of HypreParMatrix are wrapped.
   *       No preconditioner can be automatically constructed from PETSc.
   *       If @a wrap is false, the HypreParMatrix is converted into a the AIJ
   *       PETSc format, which is suitable for most preconditioning methods.
   */
  PetscKSPSolver(const mfem::HypreParMatrix& A, KSPType ksp_type = KSPCG, const std::string& prefix = std::string(),
                 bool wrap = false, bool iter_mode = false);

  /**
   * @brief Solve x = Op^{-1} b
   *
   * @param[in] b Right-hand side vector
   * @param[in,out] x Output solution vector (and initial guess if iterative mode is enabled)
   */
  virtual void Mult(const mfem::Vector& b, mfem::Vector& x) const;

  /**
   * @brief Solve x = Op^{-T} b
   *
   * @param[in] b Right-hand side vector
   * @param[in,out] x Output solution vector (and initial guess if iterative mode is enabled)
   */
  virtual void MultTranspose(const mfem::Vector& b, mfem::Vector& x) const;

  /**
   * @brief Set the underlying matrix operator to use in the solution algorithm
   *
   * @param op The matrix operator, either wrapped as a MATSHELL or converted to a PETSc MatType (often MATAIJ)
   */
  virtual void SetOperator(const mfem::Operator& op);

  /**
   * @brief Set the preconditioner for the linear solver
   *
   * @param pc The preconditioner solver
   *
   * @note For best results @a pc should be a mfem::PetscPreconditioner
   */
  virtual void SetPreconditioner(mfem::Solver& pc) { mfem::PetscLinearSolver::SetPreconditioner(pc); }
};

/**
 * @brief Wrapper for applying the PETSc algebraic multigrid preconditioner PCGAMG
 */
class PetscGAMGSolver : public mfem::PetscPreconditioner {
protected:
  /// @brief Displacement finite element space, used to construct the near null space
  mfem::ParFiniteElementSpace* fespace_ = nullptr;
  /// @brief The MPI communicator used by the vectors and matrices in the solve
  MPI_Comm comm_;

public:
  /**
   * @brief Construct a PetscGAMGSolver without an explicit operator (must be set later with SetOperator())
   *
   * @param comm The MPI communicator used by the vectors and matrices in the solve
   * @param prefix Prefix for all PETSc command line options options
   */
  PetscGAMGSolver(MPI_Comm& comm, const std::string& prefix = std::string());

  /**
   * @brief Construct a PetscGAMGSolver to approximate A^{-1}
   * @param A The mfem::PetscParMatrix for which the preconditioner is constructed
   * @param prefix Prefix for all PETSc command line options options
   */
  PetscGAMGSolver(mfem::PetscParMatrix& A, const std::string& prefix = std::string());

  /**
   * @brief Construct a PetscGAMGSolver from a generic operator
   * @param comm The MPI communicator used by the vectors and matrices in the solve
   * @param op mfem::Operator used to construct the preconditioner
   * @param prefix Prefix for all PETSc command line options options
   */
  PetscGAMGSolver(MPI_Comm comm, Operator& op, const std::string& prefix = std::string());

  /**
   * @brief Set the finite element space used to construct the rigid body near-null space
   * @param fespace Finite element space, usually corresponding to the displacement DoFs
   */
  void SetElasticityOptions(mfem::ParFiniteElementSpace* fespace);

  /**
   * @brief  Set the underlying matrix operator to use in the solution algorithm
   * @param[in] op The matrix operator for which to approximate the inverse
   */
  void SetOperator(const Operator& op);
};

}  // namespace serac::mfem_ext

#endif
