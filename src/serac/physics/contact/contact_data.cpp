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
      num_pressure_dofs_{0}
{
  tribol::initialize(mesh_.SpaceDimension(), mesh_.GetComm());
}

ContactData::~ContactData() { tribol::finalize(); }

void ContactData::addContactInteraction(int interaction_id, const std::set<int>& bdry_attr_surf1,
                                        const std::set<int>& bdry_attr_surf2, ContactOptions contact_opts)
{
  interactions_.emplace_back(interaction_id, mesh_, bdry_attr_surf1, bdry_attr_surf2, current_coords_, contact_opts);
  if (contact_opts.enforcement == ContactEnforcement::LagrangeMultiplier) {
    have_lagrange_multipliers_ = true;
    num_pressure_dofs_ += interactions_.back().numPressureDofs();
  }
}

void ContactData::update(int cycle, double time, double& dt)
{
  // This updates the redecomposed surface mesh based on the current displacement, then transfers field quantities to
  // the updated mesh.
  tribol::updateMfemParallelDecomposition();
  // This function computes forces, gaps, and Jacobian contributions based on the current field quantities. Note the
  // fields (with the exception of pressure) are stored on the redecomposed surface mesh until transferred by calling
  // forces(), mergedGaps(), etc.
  tribol::update(cycle, time, dt);
}

FiniteElementDual ContactData::forces() const
{
  FiniteElementDual f(*reference_nodes_->ParFESpace(), "contact force");
  for (const auto& interaction : interactions_) {
    f += interaction.forces();
  }
  return f;
}

mfem::Vector ContactData::mergedPressures() const
{
  mfem::Vector merged_p(numPressureDofs());
  auto         dof_offsets = pressureDofOffsets();
  for (size_t i{0}; i < interactions_.size(); ++i) {
    if (interactions_[i].getContactOptions().enforcement == ContactEnforcement::LagrangeMultiplier) {
      mfem::Vector p_interaction;
      p_interaction.MakeRef(merged_p, dof_offsets[static_cast<int>(i)],
                            dof_offsets[static_cast<int>(i) + 1] - dof_offsets[static_cast<int>(i)]);
      p_interaction.Set(1.0, interactions_[i].pressure());
    }
  }
  return merged_p;
}

mfem::Vector ContactData::mergedGaps() const
{
  mfem::Vector merged_g(numPressureDofs());
  auto         dof_offsets = pressureDofOffsets();
  for (size_t i{0}; i < interactions_.size(); ++i) {
    if (interactions_[i].getContactOptions().enforcement == ContactEnforcement::LagrangeMultiplier) {
      mfem::Vector g_interaction;
      g_interaction.MakeRef(merged_g, dof_offsets[static_cast<int>(i)],
                            dof_offsets[static_cast<int>(i) + 1] - dof_offsets[static_cast<int>(i)]);
      g_interaction.Set(1.0, interactions_[i].gaps());
    }
  }
  return merged_g;
}

std::unique_ptr<mfem::BlockOperator> ContactData::mergedJacobian() const
{
  jacobian_offsets_ = mfem::Array<int>({0, reference_nodes_->ParFESpace()->GetTrueVSize(),
                                        numPressureDofs() + reference_nodes_->ParFESpace()->GetTrueVSize()});
  // this is the BlockOperator we are returning with the following blocks:
  //  | df_(contact)/dx  df_(contact)/dp |
  //  | dg/dx            I_(inactive)    |
  // where I_(inactive) is a matrix with ones on the diagonal of inactive pressure true degrees of freedom
  auto block_J         = std::make_unique<mfem::BlockOperator>(jacobian_offsets_);
  block_J->owns_blocks = true;
  // rather than returning different blocks for each contact interaction with Lagrange multipliers, merge them all into
  // a single block
  mfem::Array2D<mfem::HypreParMatrix*> constraint_matrices(static_cast<int>(interactions_.size()), 1);

  for (size_t i{0}; i < interactions_.size(); ++i) {
    // this is the BlockOperator for one of the contact interactions
    auto interaction_J         = interactions_[i].jacobian();
    interaction_J->owns_blocks = false;  // we'll manage the ownership of the blocks on our own...
    // add the contact interaction's contribution to df_(contact)/dx (the 0, 0 block)
    if (!interaction_J->IsZeroBlock(0, 0)) {
      SLIC_ERROR_ROOT_IF(!dynamic_cast<mfem::HypreParMatrix*>(&interaction_J->GetBlock(0, 0)),
                         "Only HypreParMatrix constraint matrix blocks are currently supported.");
      if (block_J->IsZeroBlock(0, 0)) {
        block_J->SetBlock(0, 0, &interaction_J->GetBlock(0, 0));
      } else {
        if (block_J->IsZeroBlock(0, 0)) {
          block_J->SetBlock(0, 0, &interaction_J->GetBlock(0, 0));
        } else {
          block_J->SetBlock(0, 0,
                            mfem::Add(1.0, static_cast<mfem::HypreParMatrix&>(block_J->GetBlock(0, 0)), 1.0,
                                      static_cast<mfem::HypreParMatrix&>(interaction_J->GetBlock(0, 0))));
        }
        delete &interaction_J->GetBlock(0, 0);
      }
    }
    // add the contact interaction's (other) contribution to df_(contact)/dx (for penalty) or to df_(contact)/dp and
    // dg/dx (for Lagrange multipliers)
    if (!interaction_J->IsZeroBlock(1, 0)) {
      auto B = dynamic_cast<mfem::HypreParMatrix*>(&interaction_J->GetBlock(1, 0));
      SLIC_ERROR_ROOT_IF(!B, "Only HypreParMatrix constraint matrix blocks are currently supported.");
      // zero out rows not in the active set
      B->EliminateRows(interactions_[i].inactiveDofs());
      if (interactions_[i].getContactOptions().enforcement == ContactEnforcement::Penalty) {
        // compute contribution to df_(contact)/dx (the 0, 0 block) for penalty
        std::unique_ptr<mfem::HypreParMatrix> BTB(
            mfem::ParMult(std::unique_ptr<mfem::HypreParMatrix>(B->Transpose()).get(), B, true));
        delete &interaction_J->GetBlock(1, 0);
        if (block_J->IsZeroBlock(0, 0)) {
          mfem::Vector penalty(reference_nodes_->ParFESpace()->GetTrueVSize());
          penalty = interactions_[i].getContactOptions().penalty;
          BTB->ScaleRows(penalty);
          block_J->SetBlock(0, 0, BTB.release());
        } else {
          block_J->SetBlock(0, 0,
                            mfem::Add(1.0, static_cast<mfem::HypreParMatrix&>(block_J->GetBlock(0, 0)),
                                      interactions_[i].getContactOptions().penalty, *BTB));
        }
        constraint_matrices(static_cast<int>(i), 0) = nullptr;
      } else  // enforcement == ContactEnforcement::LagrangeMultiplier
      {
        // compute contribution to off-diagonal blocks for Lagrange multiplier
        constraint_matrices(static_cast<int>(i), 0) = static_cast<mfem::HypreParMatrix*>(B);
      }
      if (interaction_J->IsZeroBlock(0, 1) || !dynamic_cast<mfem::TransposeOperator*>(&interaction_J->GetBlock(0, 1))) {
        SLIC_ERROR_ROOT("Only symmetric constraint matrices are currently supported.");
      }
      delete &interaction_J->GetBlock(0, 1);
      if (!interaction_J->IsZeroBlock(1, 1)) {
        SLIC_ERROR_ROOT("Only zero-valued (1, 1) Jacobian blocks are currently supported.");
      }
    }
  }
  if (haveLagrangeMultipliers()) {
    // merge all of the contributions from all of the contact interactions
    block_J->SetBlock(1, 0, mfem::HypreParMatrixFromBlocks(constraint_matrices));
    // store the transpose explicitly (rather than as a TransposeOperator) for solvers that need HypreParMatrixs
    block_J->SetBlock(0, 1, static_cast<mfem::HypreParMatrix&>(block_J->GetBlock(1, 0)).Transpose());
    // build I_(inactive): a diagonal matrix with ones on inactive dofs and zeros elsewhere
    mfem::Array<const mfem::Array<int>*> inactive_tdofs_vector(static_cast<int>(interactions_.size()));
    int                                  inactive_tdofs_ct = 0;
    for (int i{0}; i < inactive_tdofs_vector.Size(); ++i) {
      inactive_tdofs_vector[i] = &interactions_[static_cast<size_t>(i)].inactiveDofs();
      inactive_tdofs_ct += inactive_tdofs_vector[i]->Size();
    }
    auto             dof_offsets = pressureDofOffsets();
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
    mfem::Array<int> rows(numPressureDofs() + 1);
    rows              = 0;
    inactive_tdofs_ct = 0;
    for (int i{0}; i < numPressureDofs(); ++i) {
      if (inactive_tdofs_ct < inactive_tdofs.Size() && inactive_tdofs[inactive_tdofs_ct] == i) {
        ++inactive_tdofs_ct;
      }
      rows[i + 1] = inactive_tdofs_ct;
    }
    rows.GetMemory().SetHostPtrOwner(false);
    mfem::Vector ones(inactive_tdofs_ct);
    ones = 1.0;
    ones.GetMemory().SetHostPtrOwner(false);
    mfem::SparseMatrix inactive_diag(rows.GetData(), inactive_tdofs.GetData(), ones.GetData(), numPressureDofs(),
                                     numPressureDofs(), false, false, true);
    // if the size of ones is zero, SparseMatrix creates its own memory which it
    // owns.  explicitly prevent this...
    inactive_diag.SetDataOwner(false);
    auto& block_1_0 = static_cast<mfem::HypreParMatrix&>(block_J->GetBlock(1, 0));
    auto  block_1_1 = new mfem::HypreParMatrix(block_1_0.GetComm(), block_1_0.GetGlobalNumRows(),
                                              block_1_0.GetRowStarts(), &inactive_diag);
    block_1_1->SetOwnerFlags(3, 3, 1);
    block_J->SetBlock(1, 1, block_1_1);
    // end building I_(inactive)
  }
  return block_J;
}

std::function<void(const mfem::Vector&, mfem::Vector&)> ContactData::residualFunction(
    std::function<void(const mfem::Vector&, mfem::Vector&)> orig_r)
{
  return [this, orig_r](const mfem::Vector& u, mfem::Vector& r) {
    const int disp_size = reference_nodes_->ParFESpace()->GetTrueVSize();

    mfem::Vector u_blk, p_blk;
    u_blk.MakeRef(const_cast<mfem::Vector&>(u), 0, disp_size);
    p_blk.MakeRef(const_cast<mfem::Vector&>(u), disp_size, numPressureDofs());

    mfem::Vector r_blk, g_blk;
    r_blk.MakeRef(r, 0, disp_size);
    g_blk.MakeRef(r, disp_size, numPressureDofs());

    double dt = 1.0;
    setDisplacements(u_blk);
    // we need to call update first to update gaps
    update(1, 1.0, dt);
    // with updated gaps, we can update pressure for contact interactions with penalty enforcement
    setPressures(p_blk);
    // call update again with the right pressures
    update(1, 1.0, dt);

    orig_r(u_blk, r_blk);
    r_blk += forces();

    g_blk.Set(1.0, mergedGaps());
  };
}

std::function<std::unique_ptr<mfem::BlockOperator>(const mfem::Vector&)> ContactData::jacobianFunction(
    std::function<std::unique_ptr<mfem::HypreParMatrix>(const mfem::Vector&)> orig_J) const
{
  return [this, orig_J](const mfem::Vector& u) -> std::unique_ptr<mfem::BlockOperator> {
    mfem::Vector u_blk;
    u_blk.MakeRef(const_cast<mfem::Vector&>(u), 0, reference_nodes_->ParFESpace()->GetTrueVSize());
    auto J = orig_J(u_blk);

    auto J_contact = mergedJacobian();
    if (J_contact->IsZeroBlock(0, 0)) {
      J_contact->SetBlock(0, 0, J.release());
    } else {
      J_contact->SetBlock(0, 0, mfem::Add(1.0, *J, 1.0, static_cast<mfem::HypreParMatrix&>(J_contact->GetBlock(0, 0))));
    }

    return J_contact;
  };
}

void ContactData::setPressures(const mfem::Vector& merged_pressures) const
{
  auto dof_offsets = pressureDofOffsets();
  for (size_t i{0}; i < interactions_.size(); ++i) {
    FiniteElementState p_interaction(interactions_[i].pressureSpace());
    if (interactions_[i].getContactOptions().enforcement == ContactEnforcement::LagrangeMultiplier) {
      mfem::Vector p_interaction_ref(const_cast<mfem::Vector&>(merged_pressures), dof_offsets[static_cast<int>(i)],
                                     dof_offsets[static_cast<int>(i) + 1] - dof_offsets[static_cast<int>(i)]);
      p_interaction.Set(1.0, p_interaction_ref);
    } else  // enforcement == ContactEnforcement::Penalty
    {
      p_interaction.Set(interactions_[i].getContactOptions().penalty, interactions_[i].gaps());
    }
    for (auto dof : interactions_[i].inactiveDofs()) {
      p_interaction[dof] = 0.0;
    }
    interactions_[i].setPressure(p_interaction);
  }
}

void ContactData::setDisplacements(const mfem::Vector& u)
{
  reference_nodes_->ParFESpace()->GetProlongationMatrix()->Mult(u, current_coords_);
  current_coords_ += *reference_nodes_;
}

mfem::Array<int> ContactData::pressureDofOffsets() const
{
  mfem::Array<int> dof_offsets(static_cast<int>(interactions_.size()) + 1);
  dof_offsets = 0;
  for (size_t i{0}; i < interactions_.size(); ++i) {
    dof_offsets[static_cast<int>(i + 1)] = dof_offsets[static_cast<int>(i)] + interactions_[i].numPressureDofs();
  }
  return dof_offsets;
}

#else

namespace serac {

ContactData::ContactData([[maybe_unused]] const mfem::ParMesh& mesh)
    : have_lagrange_multipliers_{false}, num_pressure_true_dofs_{0}
{
}

ContactData::~ContactData() {}

void ContactData::addContactInteraction([[maybe_unused]] int                  interaction_id,
                                        [[maybe_unused]] const std::set<int>& bdry_attr_surf1,
                                        [[maybe_unused]] const std::set<int>& bdry_attr_surf2,
                                        [[maybe_unused]] ContactOptions       contact_opts)
{
  SLIC_WARNING_ROOT("Serac built without Tribol support. No contact interaction will be added.");
}

bool ContactData::haveContactInteractions() const { return false; }

void ContactData::update([[maybe_unused]] int cycle, [[maybe_unused]] double time, [[maybe_unused]] double& dt) {}

FiniteElementDual ContactData::forces() const
{
  FiniteElementDual f(*reference_nodes_->ParFESpace(), "contact force");
  f = 0.0;
  return f;
}

mfem::Vector ContactData::mergedPressures() const { return mfem::Vector(); }

mfem::Vector ContactData::mergedGaps() const { return mfem::Vector(); }

std::unique_ptr<mfem::BlockOperator> ContactData::mergedJacobian() const
{
  jacobian_offsets_ = mfem::Array<int>(
      {0, reference_nodes_->ParFESpace()->GetTrueVSize(), reference_nodes_->ParFESpace()->GetTrueVSize()});
  return std::make_unique<mfem::BlockOperator>(jacobian_offsets_);
}

std::function<void(const mfem::Vector&, mfem::Vector&)> ContactData::residualFunction(
    std::function<void(const mfem::Vector&, mfem::Vector&)> orig_r)
{
  return orig_r;
}

std::function<std::unique_ptr<mfem::BlockOperator>(const mfem::Vector&)> ContactData::jacobianFunction(
    std::function<std::unique_ptr<mfem::HypreParMatrix>(const mfem::Vector&)> orig_J) const
{
  return [orig_J](const mfem::Vector& u) -> std::unique_ptr<mfem::BlockOperator> {
    auto J = orig_J(u);

    auto J_contact         = std::make_unique<mfem::BlockOperator>(jacobian_offsets_);
    J_contact->owns_blocks = true;
    J_contact->SetBlock(0, 0, J.release());

    return J_contact;
  };
}

void ContactData::setPressures([[maybe_unused]] const mfem::Vector& true_pressures) const {}

void ContactData::setDisplacements([[maybe_unused]] const mfem::Vector& true_displacement) {}

mfem::Array<int> ContactData::pressureDofOffsets() const { return mfem::Array<int>(); }

#endif

}  // namespace serac
