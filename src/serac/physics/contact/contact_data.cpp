// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/contact/contact_data.hpp"

#include "axom/slic.hpp"

#ifdef SERAC_USE_TRIBOL
#include "tribol/interface/tribol.hpp"
#include "tribol/interface/mfem_tribol.hpp"

namespace serac {

ContactData::ContactData(const mfem::ParMesh& mesh)
    : mesh_{mesh},
      reference_nodes_{dynamic_cast<const mfem::ParGridFunction*>(mesh.GetNodes())},
      current_coords_{*reference_nodes_},
      have_lagrange_multipliers_{false},
      num_pressure_true_dofs_{0}
{
  tribol::initialize(mesh_.SpaceDimension(), mesh_.GetComm());
}

ContactData::~ContactData()
{
  tribol::finalize();
}

void ContactData::addContactPair(int pair_id, const std::set<int>& bdry_attr_surf1, const std::set<int>& bdry_attr_surf2,
                                 ContactOptions contact_opts)
{
  pairs_.emplace_back(pair_id, mesh_, bdry_attr_surf1, bdry_attr_surf2, current_coords_, contact_opts);
  if (contact_opts.enforcement == ContactEnforcement::LagrangeMultiplier) {
    have_lagrange_multipliers_ = true;
    num_pressure_true_dofs_ += pairs_.back().numPressureTrueDofs();
  }
}

bool ContactData::haveContactPairs() const
{
  return !pairs_.empty();
}

void ContactData::update(int cycle, double time, double& dt)
{
  tribol::updateMfemParallelDecomposition();
  tribol::update(cycle, time, dt);
}

mfem::Vector ContactData::trueContactForces() const
{
  mfem::Vector f_true(reference_nodes_->ParFESpace()->GetTrueVSize());
  mfem::Vector f(reference_nodes_->ParFESpace()->GetVSize());
  f = 0.0;
  for (const auto& pair : pairs_) {
    f += pair.contactForces();
  }
  // NOTE: forces are considered a dual field in MFEM and the correct operator
  // here for MFEM dual fields is P^T.  However, Tribol stores forces similar to
  // a ParGridFunction -- all shared dofs are equal; therefore, the operator
  // that returns the desired values is R
  reference_nodes_->ParFESpace()->GetRestrictionMatrix()->Mult(f, f_true);
  return f_true;
}

mfem::Vector ContactData::truePressures() const
{
  mfem::Vector p_true(numPressureTrueDofs());
  auto dof_offsets = pressureTrueDofOffsets();
  for (size_t i{0}; i < pairs_.size(); ++i) {
    if (pairs_[i].getContactOptions().enforcement == ContactEnforcement::LagrangeMultiplier) {
      mfem::Vector p_pair_true;
      p_pair_true.MakeRef(p_true, dof_offsets[static_cast<int>(i)],
                          dof_offsets[static_cast<int>(i) + 1] - dof_offsets[static_cast<int>(i)]);
      pairs_[i].pressure().ParFESpace()->GetRestrictionMatrix()->Mult(pairs_[i].pressure(),
                                                                      p_pair_true);
    }
  }
  return p_true;
}

mfem::Vector ContactData::trueGaps() const
{
  mfem::Vector g_true(numPressureTrueDofs());
  auto dof_offsets = pressureTrueDofOffsets();
  for (size_t i{0}; i < pairs_.size(); ++i) {
    if (pairs_[i].getContactOptions().enforcement == ContactEnforcement::LagrangeMultiplier) {
      mfem::Vector g_pair_true;
      g_pair_true.MakeRef(g_true, dof_offsets[static_cast<int>(i)],
                          dof_offsets[static_cast<int>(i) + 1] - dof_offsets[static_cast<int>(i)]);
      // NOTE: gaps are considered a dual field in MFEM and the correct operator
      // here for MFEM dual fields is P^T.  However, Tribol stores gaps similar
      // to a ParGridFunction -- all shared dofs are equal; therefore, the
      // operator that returns the desired values is R
      pairs_[i].pressure().ParFESpace()->GetRestrictionMatrix()->Mult(pairs_[i].gaps(),
                                                                      g_pair_true);
    }
  }
  return g_true;
}

std::unique_ptr<mfem::BlockOperator> ContactData::contactJacobian() const
{
  jacobian_offsets_    = mfem::Array<int>({0, reference_nodes_->ParFESpace()->GetTrueVSize(),
                                        numPressureTrueDofs() + reference_nodes_->ParFESpace()->GetTrueVSize()});
  auto block_J         = std::make_unique<mfem::BlockOperator>(jacobian_offsets_);
  block_J->owns_blocks = true;
  mfem::Array2D<mfem::HypreParMatrix*> constraint_matrices(static_cast<int>(pairs_.size()), 1);
  
  for (size_t i{0}; i < pairs_.size(); ++i) {
    auto pair_J         = tribol::getMfemBlockJacobian(pairs_[i].getPairId());
    pair_J->owns_blocks = false;
    if (!pair_J->IsZeroBlock(0, 0)) {
      SLIC_ERROR_ROOT_IF(!dynamic_cast<mfem::HypreParMatrix*>(&pair_J->GetBlock(0, 0)),
                         "Only HypreParMatrix constraint matrix blocks are currently supported.");
      if (block_J->IsZeroBlock(0, 0)) {
        block_J->SetBlock(0, 0, &pair_J->GetBlock(0, 0));
      } else {
        if (block_J->IsZeroBlock(0, 0)) {
          block_J->SetBlock(0, 0, &pair_J->GetBlock(0, 0));
        } else {
          block_J->SetBlock(0, 0,
                            mfem::Add(1.0, static_cast<mfem::HypreParMatrix&>(block_J->GetBlock(0, 0)), 1.0,
                                      static_cast<mfem::HypreParMatrix&>(pair_J->GetBlock(0, 0))));
        }
        delete &pair_J->GetBlock(0, 0);
      }
    }
    if (!pair_J->IsZeroBlock(1, 0)) {
      auto B = dynamic_cast<mfem::HypreParMatrix*>(&pair_J->GetBlock(1, 0));
      SLIC_ERROR_ROOT_IF(!B, "Only HypreParMatrix constraint matrix blocks are currently supported.");
      mfem::Vector active_rows(B->Height());
      active_rows = 1.0;
      for (auto inactive_dof : pairs_[i].inactiveTrueDofs())
      {
        active_rows[inactive_dof] = 0.0;
      }
      B->ScaleRows(active_rows);
      if (pairs_[i].getContactOptions().enforcement == ContactEnforcement::Penalty) {
        std::unique_ptr<mfem::HypreParMatrix> BTB(
            mfem::ParMult(std::unique_ptr<mfem::HypreParMatrix>(B->Transpose()).get(), B, true));
        delete &pair_J->GetBlock(1, 0);
        if (block_J->IsZeroBlock(0, 0)) {
          mfem::Vector penalty(reference_nodes_->ParFESpace()->GetTrueVSize());
          penalty = pairs_[i].getContactOptions().penalty;
          BTB->ScaleRows(penalty);
          block_J->SetBlock(0, 0, BTB.release());
        } else {
          block_J->SetBlock(0, 0,
                            mfem::Add(1.0, static_cast<mfem::HypreParMatrix&>(block_J->GetBlock(0, 0)),
                                      pairs_[i].getContactOptions().penalty, *BTB));
        }
        constraint_matrices(static_cast<int>(i), 0) = nullptr;
      } else  // enforcement == ContactEnforcement::LagrangeMultiplier
      {
        constraint_matrices(static_cast<int>(i), 0) = static_cast<mfem::HypreParMatrix*>(B);
      }
      if (pair_J->IsZeroBlock(0, 1) || !dynamic_cast<mfem::TransposeOperator*>(&pair_J->GetBlock(0, 1))) {
        SLIC_ERROR_ROOT("Only symmetric constraint matrices are currently supported.");
      }
      delete &pair_J->GetBlock(0, 1);
      if (!pair_J->IsZeroBlock(1, 1)) {
        SLIC_ERROR_ROOT("Only zero-valued (1, 1) Jacobian blocks are currently supported.");
      }
    }
  }
  if (haveLagrangeMultipliers()) {
    block_J->SetBlock(1, 0, mfem::HypreParMatrixFromBlocks(constraint_matrices));
    block_J->SetBlock(0, 1, static_cast<mfem::HypreParMatrix&>(block_J->GetBlock(1, 0)).Transpose());
    // build diagonal matrix with ones on inactive dofs
    mfem::Array<const mfem::Array<int>*> inactive_tdofs_vector(static_cast<int>(pairs_.size()));
    int inactive_tdofs_ct = 0;
    for (int i{0}; i < inactive_tdofs_vector.Size(); ++i) {
      if (pairs_[static_cast<size_t>(i)].numPressureTrueDofs() > 0) {
        inactive_tdofs_vector[i] = &pairs_[static_cast<size_t>(i)].inactiveTrueDofs();
        inactive_tdofs_ct += inactive_tdofs_vector[i]->Size();
      }
    }
    auto dof_offsets = pressureTrueDofOffsets();
    mfem::Array<int> inactive_tdofs(inactive_tdofs_ct);
    inactive_tdofs_ct = 0;
    for (int i{0}; i < inactive_tdofs_vector.Size(); ++i) {
      if (inactive_tdofs_vector[i]) {
        for (int d{0}; d < inactive_tdofs_vector[i]->Size(); ++d) {
          inactive_tdofs[d + inactive_tdofs_ct] = (*inactive_tdofs_vector[i])[d] + dof_offsets[i];
        }
        inactive_tdofs_ct += inactive_tdofs_vector[i]->Size();
      }
    }
    inactive_tdofs.GetMemory().SetHostPtrOwner(false);
    mfem::Array<int> rows(numPressureTrueDofs() + 1);
    rows = 0;
    inactive_tdofs_ct = 0;
    for (int i{0}; i < numPressureTrueDofs(); ++i)
    {
      if (inactive_tdofs_ct < inactive_tdofs.Size() && inactive_tdofs[inactive_tdofs_ct] == i)
      {
        ++inactive_tdofs_ct;
      }
      rows[i + 1] = inactive_tdofs_ct;
    }
    rows.GetMemory().SetHostPtrOwner(false);
    mfem::Vector ones(inactive_tdofs_ct);
    ones = 1.0;
    ones.GetMemory().SetHostPtrOwner(false);
    mfem::SparseMatrix inactive_diag(rows.GetData(), inactive_tdofs.GetData(), ones.GetData(), 
                                     numPressureTrueDofs(), numPressureTrueDofs(), false, false, true);
    // if the size of ones is zero, SparseMatrix creates its own memory which it
    // owns.  explicitly prevent this...
    inactive_diag.SetDataOwner(false);
    auto& block_1_0 = static_cast<mfem::HypreParMatrix&>(block_J->GetBlock(1, 0));
    auto block_1_1 = new mfem::HypreParMatrix(block_1_0.GetComm(), block_1_0.GetGlobalNumRows(), 
                                              block_1_0.GetRowStarts(), &inactive_diag);
    block_1_1->SetOwnerFlags(3, 3, 1);
    block_J->SetBlock(1, 1, block_1_1);
  }
  return block_J;
}

void ContactData::setPressures(const mfem::Vector& true_pressures) const
{
  auto dof_offsets = pressureTrueDofOffsets();
  for (size_t i{0}; i < pairs_.size(); ++i) {
    auto& pressure = pairs_[i].pressure();
    if (pairs_[i].getContactOptions().enforcement == ContactEnforcement::LagrangeMultiplier) {
      mfem::Vector p_pair_true;
      p_pair_true.MakeRef(const_cast<mfem::Vector&>(true_pressures), dof_offsets[static_cast<int>(i)],
                          dof_offsets[static_cast<int>(i) + 1] - dof_offsets[static_cast<int>(i)]);
      pressure.ParFESpace()->GetProlongationMatrix()->Mult(p_pair_true, pairs_[i].pressure());
    } else  // enforcement == ContactEnforcement::Penalty
    {
      pressure.Set(pairs_[i].getContactOptions().penalty, pairs_[i].gaps());
    }
    for (auto dof : pairs_[i].inactiveTrueDofs())
    {
      pressure[dof] = 0.0;
    }
  }
}

void ContactData::setDisplacements(const mfem::Vector& true_displacement)
{
  reference_nodes_->ParFESpace()->GetProlongationMatrix()->Mult(true_displacement, current_coords_);
  current_coords_ += *reference_nodes_;
}

mfem::Array<int> ContactData::pressureTrueDofOffsets() const
{
  mfem::Array<int> dof_offsets(static_cast<int>(pairs_.size()) + 1);
  dof_offsets = 0;
  for (size_t i{0}; i < pairs_.size(); ++i) {
    dof_offsets[static_cast<int>(i + 1)] = dof_offsets[static_cast<int>(i)] + pairs_[i].numPressureTrueDofs();
  }
  return dof_offsets;
}

#else

namespace serac {

ContactData::ContactData(const mfem::ParMesh& mesh)
    : mesh_{mesh}
{}

ContactData::~ContactData()
{}

void ContactData::addContactPair([[maybe_unused]] int pair_id, [[maybe_unused]] const std::set<int>& bdry_attr_surf1, 
                                 [[maybe_unused]] const std::set<int>& bdry_attr_surf2,
                                 [[maybe_unused]] ContactOptions       contact_opts)
{
  SLIC_WARNING_ROOT("Serac built without Tribol support. No contact pair will be added.");
}

bool ContactData::haveContactPairs() const
{
  return false;
}

void ContactData::update([[maybe_unused]] int cycle, [[maybe_unused]] double time, [[maybe_unused]] double& dt, 
                         [[maybe_unused]] bool update_redecomp)
{}

mfem::Vector ContactData::trueContactForces() const
{
  mfem::Vector f_true(reference_nodes_->ParFESpace()->GetTrueVSize());
  f_true = 0.0;
  return f_true;
}

mfem::Vector ContactData::truePressures() const
{
  return mfem::Vector();
}

mfem::Vector ContactData::trueGaps() const
{
  return mfem::Vector();
}

std::unique_ptr<mfem::BlockOperator> ContactData::contactJacobian() const
{
  jacobian_offsets_ = mfem::Array<int>({0, reference_nodes_->ParFESpace()->GetTrueVSize(), 
                                        reference_nodes_->ParFESpace()->GetTrueVSize()});
  return std::make_unique<mfem::BlockOperator>(jacobian_offsets_);
}

void ContactData::setPressures([[maybe_unused]] const mfem::Vector& true_pressures) const
{}

void ContactData::setDisplacements([[maybe_unused]] const mfem::Vector& true_displacement)
{}

mfem::Array<int> ContactData::pressureTrueDofOffsets() const
{
  return mfem::Array<int>();
}

#endif

}  // namespace serac
