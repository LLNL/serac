// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/infrastructure/petsc_ext.hpp"
#include "serac/infrastructure/logger.hpp"

#ifdef MFEM_USE_PETSC
namespace serac::mfem_ext {

// PetscKSPSolver methods

PetscKSPSolver::PetscKSPSolver(MPI_Comm comm, KSPType ksp_type, const std::string& prefix, bool wrap, bool iter_mode)
    : mfem::IterativeSolver(comm), mfem::PetscLinearSolver(comm, prefix, wrap, iter_mode), comm_(comm), wrap_(wrap)
{
  KSPSetType(*this, ksp_type);
  Customize();
}

PetscKSPSolver::PetscKSPSolver(const mfem::PetscParMatrix& A, KSPType ksp_type, const std::string& prefix,
                               bool iter_mode)
    : mfem::IterativeSolver(A.GetComm()),
      mfem::PetscLinearSolver(A, prefix, iter_mode),
      comm_(A.GetComm()),
      wrap_(false)
{
  KSPSetType(*this, ksp_type);
  Customize();
}

PetscKSPSolver::PetscKSPSolver(const mfem::HypreParMatrix& A, KSPType ksp_type, const std::string& prefix, bool wrap,
                               bool iter_mode)
    : mfem::IterativeSolver(A.GetComm()),
      mfem::PetscLinearSolver(A, wrap, prefix, iter_mode),
      comm_(A.GetComm()),
      wrap_(wrap)
{
  KSPSetType(*this, ksp_type);
  Customize();
}

void PetscKSPSolver::Mult(const mfem::Vector& b, mfem::Vector& x) const { mfem::PetscLinearSolver::Mult(b, x); }

void PetscKSPSolver::MultTranspose(const mfem::Vector& b, mfem::Vector& x) const
{
  mfem::PetscLinearSolver::MultTranspose(b, x);
}

void PetscKSPSolver::SetOperator(const mfem::Operator& op)
{
  const mfem::HypreParMatrix* hA = dynamic_cast<const mfem::HypreParMatrix*>(&op);
  mfem::PetscParMatrix*       pA = const_cast<mfem::PetscParMatrix*>(dynamic_cast<const mfem::PetscParMatrix*>(&op));
  const mfem::Operator*       oA = dynamic_cast<const mfem::Operator*>(&op);

  // delete existing matrix, if created
  if (pA_) delete pA_;
  // update base classes: Operator, Solver, PetscLinearSolver
  if (!pA) {
    if (hA) {
      // Create MATSHELL object or convert into a format suitable to construct preconditioners
      if constexpr (PETSC_HAVE_HYPRE) {
        pA = new mfem::PetscParMatrix(hA, wrap_ ? PETSC_MATSHELL : PETSC_MATAIJ);
      } else {
        pA = new mfem::PetscParMatrix(hA, wrap_ ? PETSC_MATSHELL : PETSC_MATAIJ);
      }
    } else if (oA)  // fallback to general operator
    {
      // Create MATSHELL or MATNEST (if oA is a BlockOperator) object
      // If oA is a BlockOperator, Operator::Type is relevant to the subblocks
      pA = new mfem::PetscParMatrix(comm_, oA, wrap_ ? PETSC_MATSHELL : PETSC_MATAIJ);
    }
    pA_ = pA;
  }
  MFEM_VERIFY(pA, "Unsupported operation!");

  // Set operators into PETSc KSP
  KSP ksp = *this;
  Mat A   = pA->ReleaseMat(false);
  if (operatorset) {
    Mat      C;
    PetscInt nheight, nwidth, oheight, owidth;

    PetscCallAbort(comm_, KSPGetOperators(ksp, &C, NULL));
    PetscCallAbort(comm_, MatGetSize(A, &nheight, &nwidth));
    PetscCallAbort(comm_, MatGetSize(C, &oheight, &owidth));
    if (nheight != oheight || nwidth != owidth) {
      // reinit without destroying the KSP
      // communicator remains the same
      PetscCallAbort(comm_, KSPReset(ksp));
      delete X;
      delete B;
      X = B = NULL;
    }
  }
  PetscCallAbort(comm_, KSPSetOperators(ksp, A, A));

  // Update PetscSolver
  operatorset = true;

  // Update the Operator fields.
  IterativeSolver::height   = pA->Height();
  PetscLinearSolver::height = pA->Height();
  IterativeSolver::width    = pA->Width();
  PetscLinearSolver::width  = pA->Width();

  mfem::IterativeSolver::SetOperator(*pA);
}

// PetscGAMGSolver methods

PetscGAMGSolver::PetscGAMGSolver(MPI_Comm& comm, const std::string& prefix)
    : mfem::PetscPreconditioner(comm, prefix), comm_(comm)
{
  PetscCallVoid(PCSetType(*this, PCGAMG));
  Customize();
}

PetscGAMGSolver::PetscGAMGSolver(mfem::PetscParMatrix& A, const std::string& prefix)
    : mfem::PetscPreconditioner(A, prefix), comm_(A.GetComm())
{
  PetscCallVoid(PCSetType(*this, PCGAMG));
  Customize();
}

PetscGAMGSolver::PetscGAMGSolver(MPI_Comm comm, Operator& op, const std::string& prefix)
    : mfem::PetscPreconditioner(comm, op, prefix), comm_(comm)
{
  PetscCallVoid(PCSetType(*this, PCGAMG));
  Customize();
}

static void func_coords(const mfem::Vector& x, mfem::Vector& y) { y = x; }

void PetscGAMGSolver::SetElasticityOptions(mfem::ParFiniteElementSpace* fespace)
{
  using namespace mfem;
  // Save the finite element space to support multiple calls to SetOperator()
  fespace_ = fespace;

  // get PETSc object
  PC pc = *this;

  PetscBool is_op_set;
  PetscCallAbort(comm_, PCGetOperatorsSet(pc, nullptr, &is_op_set));
  if (!is_op_set) return;

  Mat pA;
  PetscCallAbort(comm_, PCGetOperators(pc, NULL, &pA));

  PetscBool ismatis, ismataij;
  bool      has_local_mat;
  PetscCallAbort(comm_, PetscObjectTypeCompare(reinterpret_cast<PetscObject>(pA), MATIS, &ismatis));
  PetscCallAbort(comm_, PetscObjectTypeCompare(reinterpret_cast<PetscObject>(pA), MATAIJ, &ismataij));
  has_local_mat = ismatis || ismataij;

  if (fespace_) {
    PetscInt sdim = fespace_->GetParMesh()->SpaceDimension();
    int      vdim = fespace_->GetVDim();

    // Ideally, the block size should be set at matrix creation
    // but the MFEM assembly does not allow to do so
    PetscCallAbort(comm_, MatSetBlockSize(pA, vdim));

    // coordinates
    const FiniteElementCollection* fec     = fespace_->FEColl();
    bool                           h1space = dynamic_cast<const H1_FECollection*>(fec);
    if (h1space) {
      ParFiniteElementSpace* fespace_coords = fespace_;

      sdim = fespace_->GetParMesh()->SpaceDimension();
      if (vdim != sdim || fespace_->GetOrdering() != Ordering::byVDIM) {
        fespace_coords = new ParFiniteElementSpace(fespace_->GetParMesh(), fec, sdim, Ordering::byVDIM);
      }
      VectorFunctionCoefficient coeff_coords(sdim, func_coords);
      ParGridFunction           gf_coords(fespace_coords);
      gf_coords.ProjectCoefficient(coeff_coords);
      int             num_nodes   = fespace_->GetNDofs();
      HypreParVector* hvec_coords = gf_coords.ParallelProject();
      auto data_coords = const_cast<PetscScalar*>(mfem::Read(hvec_coords->GetMemory(), hvec_coords->Size(), false));
      PetscCallAbort(comm_, PCSetCoordinates(*this, sdim, num_nodes, data_coords));

      MatNullSpace nnsp;
      Vec          pvec_coords;

      PetscCallAbort(comm_, VecCreateMPIWithArray(comm_, sdim, hvec_coords->Size(), hvec_coords->GlobalSize(),
                                                  data_coords, &pvec_coords));
      PetscCallAbort(comm_, MatGetNearNullSpace(pA, &nnsp));
      PetscCallAbort(comm_, MatNullSpaceCreateRigidBody(pvec_coords, &nnsp));
      PetscCallAbort(comm_, MatSetNearNullSpace(pA, nnsp));
      PetscCallAbort(comm_, MatNullSpaceDestroy(&nnsp));

      // likely elasticity -> we attach rigid-body modes as near-null space information to the local matrices
      // and to the global matrix
      if (vdim == sdim) {
        if (has_local_mat) {
          Mat                    lA = nullptr;
          Vec                    lvec_coords;
          ISLocalToGlobalMapping l2g;
          PetscSF                sf;
          PetscLayout            rmap;
          const PetscInt*        gidxs;
          PetscInt               nleaves;

          if (ismatis) {
            PetscCallAbort(comm_, MatISGetLocalMat(pA, &lA));
          } else if (ismataij) {
            PetscCallAbort(comm_, MatAIJGetLocalMat(pA, &lA));
          } else {
            SLIC_ERROR_ROOT("Unsupported mat type.");
          }
          PetscCallAbort(comm_, MatCreateVecs(lA, &lvec_coords, NULL));
          PetscCallAbort(comm_, VecSetBlockSize(lvec_coords, sdim));
          PetscCallAbort(comm_, MatGetLocalToGlobalMapping(pA, &l2g, NULL));
          PetscCallAbort(comm_, MatGetLayouts(pA, &rmap, NULL));
          PetscCallAbort(comm_, PetscSFCreate(comm_, &sf));
          PetscCallAbort(comm_, ISLocalToGlobalMappingGetIndices(l2g, &gidxs));
          PetscCallAbort(comm_, ISLocalToGlobalMappingGetSize(l2g, &nleaves));
          PetscCallAbort(comm_, PetscSFSetGraphLayout(sf, rmap, nleaves, NULL, PETSC_OWN_POINTER, gidxs));
          PetscCallAbort(comm_, ISLocalToGlobalMappingRestoreIndices(l2g, &gidxs));
          {
            PetscReal* garray;
            PetscReal* larray;

            PetscCallAbort(comm_, VecGetArray(pvec_coords, &garray));
            PetscCallAbort(comm_, VecGetArray(lvec_coords, &larray));
#if PETSC_VERSION_LT(3, 15, 0)
            PetscCallAbort(comm_, PetscSFBcastBegin(sf, MPIU_SCALAR, garray, larray));
            PetscCallAbort(comm_, PetscSFBcastEnd(sf, MPIU_SCALAR, garray, larray));
#else
            PetscCallAbort(comm_, PetscSFBcastBegin(sf, MPIU_SCALAR, garray, larray, MPI_REPLACE));
            PetscCallAbort(comm_, PetscSFBcastEnd(sf, MPIU_SCALAR, garray, larray, MPI_REPLACE));
#endif
            PetscCallAbort(comm_, VecRestoreArray(pvec_coords, &garray));
            PetscCallAbort(comm_, VecRestoreArray(lvec_coords, &larray));
          }
          PetscCallAbort(comm_, MatNullSpaceCreateRigidBody(lvec_coords, &nnsp));
          PetscCallAbort(comm_, VecDestroy(&lvec_coords));
          PetscCallAbort(comm_, MatSetNearNullSpace(lA, nnsp));
          PetscCallAbort(comm_, MatNullSpaceDestroy(&nnsp));
          PetscCallAbort(comm_, PetscSFDestroy(&sf));
        }
        PetscCallAbort(comm_, VecDestroy(&pvec_coords));
      }
      if (fespace_coords != fespace_) {
        delete fespace_coords;
      }
      delete hvec_coords;
    }
  }

  MatNullSpace nnsp;
  PetscCallAbort(comm_, MatGetNearNullSpace(pA, &nnsp));
  SLIC_WARNING_ROOT_IF(!nnsp, "Global near null space was not set successfully, expect slow (or no) convergence.");
}

void PetscGAMGSolver::SetOperator(const Operator& op)
{
  // Update parent class
  PetscPreconditioner::SetOperator(op);
  // Set rigid body near null space
  SLIC_WARNING_ROOT_IF(
      fespace_ == nullptr,
      "Displacement FE space not set with PetscGAMGSolver::SetElasticityOptions, expect slow (or no) convergence.");
  if (fespace_) {
    SetElasticityOptions(fespace_);
  }
}

}  // namespace serac::mfem_ext

#endif
