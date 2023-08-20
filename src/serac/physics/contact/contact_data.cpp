// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/contact/contact_data.hpp"

#include "axom/slic.hpp"

#include "serac/physics/contact/contact_config.hpp"

#ifdef SERAC_USE_TRIBOL
#include "tribol/interface/tribol.hpp"
#include "tribol/interface/mfem_tribol.hpp"
#endif

namespace serac {

ContactData::ContactData(const mfem::ParMesh& mesh)
: mesh_ { mesh },
  reference_nodes_ { dynamic_cast<const mfem::ParGridFunction*>(mesh.GetNodes()) },
  current_coords_ { *reference_nodes_ },
  num_pressure_true_dofs_ { 0 }
{
#ifdef SERAC_USE_TRIBOL
  tribol::initialize(mesh_.SpaceDimension(), mesh_.GetComm());
#endif
}

ContactData::~ContactData()
{
#ifdef SERAC_USE_TRIBOL
  tribol::finalize();
#endif
}

void ContactData::addContactPair(
  [[maybe_unused]] int pair_id,
  [[maybe_unused]] const std::set<int>& bdry_attr_surf1,
  [[maybe_unused]] const std::set<int>& bdry_attr_surf2,
  [[maybe_unused]] ContactOptions contact_opts
)
{
#ifdef SERAC_USE_TRIBOL
  contactPairs().emplace_back(
    pair_id,
    mesh_,
    bdry_attr_surf1,
    bdry_attr_surf2,
    current_coords_,
    contact_opts
  );
  if (contact_opts.enforcement == ContactEnforcement::LagrangeMultiplier)
  {
    num_pressure_true_dofs_ += contactPairs().back().numTruePressureDofs();
  }
  
#else
  SLIC_WARNING_ROOT("Serac not built with Tribol. No contact pair added.");
#endif
}

void ContactData::update(
  [[maybe_unused]] int cycle, 
  [[maybe_unused]] double time, 
  [[maybe_unused]] double& dt,
  [[maybe_unused]] bool update_redecomp
)
{
#ifdef SERAC_USE_TRIBOL
  if (update_redecomp)
  {
    tribol::updateMfemParallelDecomposition();
  }
  tribol::update(cycle, time, dt);
#endif
}

mfem::Vector ContactData::trueContactForces() const
{
  mfem::Vector f_true(reference_nodes_->ParFESpace()->GetTrueVSize());
#ifdef SERAC_USE_TRIBOL
  mfem::Vector f(reference_nodes_->ParFESpace()->GetVSize());
  f = 0.0;
  for (const auto& pair : contactPairs())
  {
    f += pair.contactForces();
  }
  reference_nodes_->ParFESpace()->GetProlongationMatrix()->MultTranspose(f, f_true);
#else
  f_true = 0.0;
#endif
  return f_true;
}

mfem::Vector ContactData::truePressures() const
{
  mfem::Vector p_true(numPressureTrueDofs());
#ifdef SERAC_USE_TRIBOL
  auto dof_offsets = pressureTrueDofOffsets();
  for (size_t i{0}; i < contactPairs().size(); ++i)
  {
    if (contactPairs()[i].getContactOptions().enforcement == ContactEnforcement::LagrangeMultiplier)
    {
      mfem::Vector p_pair_true;
      p_pair_true.MakeRef(
        p_true, 
        dof_offsets[static_cast<int>(i)],
        dof_offsets[static_cast<int>(i)+1] - dof_offsets[static_cast<int>(i)]
      );
      contactPairs()[i].pressure().ParFESpace()->GetProlongationMatrix()->MultTranspose(
        contactPairs()[i].pressure(),
        p_pair_true
      );
    }
  }
#endif
  return p_true;
}

mfem::Vector ContactData::trueGaps() const
{
  mfem::Vector g_true(numPressureTrueDofs());
#ifdef SERAC_USE_TRIBOL
  auto dof_offsets = pressureTrueDofOffsets();
  for (size_t i{0}; i < contactPairs().size(); ++i)
  {
    if (contactPairs()[i].getContactOptions().enforcement == ContactEnforcement::LagrangeMultiplier)
    {
      mfem::Vector g_pair_true;
      g_pair_true.MakeRef(
        g_true, 
        dof_offsets[static_cast<int>(i)],
        dof_offsets[static_cast<int>(i)+1] - dof_offsets[static_cast<int>(i)]
      );
      contactPairs()[i].pressure().ParFESpace()->GetProlongationMatrix()->MultTranspose(
        contactPairs()[i].gaps(),
        g_pair_true
      );
    }
  }
#endif
  return g_true;
}

std::unique_ptr<mfem::BlockOperator> ContactData::contactJacobian() const
{
  jacobian_offsets_ = mfem::Array<int>({
    0, 
    reference_nodes_->ParFESpace()->GetTrueVSize(), 
    numPressureTrueDofs() + reference_nodes_->ParFESpace()->GetTrueVSize()
  });
  auto block_J = std::make_unique<mfem::BlockOperator>(jacobian_offsets_);
  block_J->owns_blocks = true;
#ifdef SERAC_USE_TRIBOL
  mfem::Array2D<mfem::HypreParMatrix*> constraint_matrices(
    static_cast<int>(contactPairs().size()),
    1
  );
  for (size_t i{0}; i < contactPairs().size(); ++i)
  {
    auto pair_J = tribol::getMfemBlockJacobian(contactPairs()[i].getPairId());
    pair_J->owns_blocks = false;
    if (!pair_J->IsZeroBlock(0, 0))
    {
      SLIC_ERROR_ROOT_IF(
        !dynamic_cast<mfem::HypreParMatrix*>(&pair_J->GetBlock(0, 0)),
        "Only HypreParMatrix constraint matrix blocks are currently supported."
      );
      if (block_J->IsZeroBlock(0, 0))
      {
        block_J->SetBlock(0, 0, &pair_J->GetBlock(0, 0));
      }
      else
      {
        if (block_J->IsZeroBlock(0, 0))
        {
          block_J->SetBlock(0, 0, &pair_J->GetBlock(0, 0));
        }
        else
        {
          block_J->SetBlock(0, 0, mfem::Add(
            1.0, static_cast<mfem::HypreParMatrix&>(block_J->GetBlock(0, 0)),
            1.0, static_cast<mfem::HypreParMatrix&>(pair_J->GetBlock(0, 0))
          ));
        }
        delete &pair_J->GetBlock(0, 0);
      }
    }
    if (!pair_J->IsZeroBlock(1, 0))
    {
      SLIC_ERROR_ROOT_IF(
        !dynamic_cast<mfem::HypreParMatrix*>(&pair_J->GetBlock(1, 0)),
        "Only HypreParMatrix constraint matrix blocks are currently supported."
      );
      if (contactPairs()[i].getContactOptions().enforcement == ContactEnforcement::Penalty)
      {
        std::unique_ptr<mfem::HypreParMatrix> BTB(
          mfem::ParMult(
            std::unique_ptr<mfem::HypreParMatrix>(
              static_cast<mfem::HypreParMatrix&>(pair_J->GetBlock(1, 0)).Transpose()
            ).get(),
            &static_cast<mfem::HypreParMatrix&>(pair_J->GetBlock(1, 0)),
            true
          )
        );
        delete &pair_J->GetBlock(1, 0);
        if (block_J->IsZeroBlock(0, 0))
        {
          mfem::Vector penalty(reference_nodes_->ParFESpace()->GetTrueVSize());
          penalty = contactPairs()[i].getContactOptions().penalty;
          BTB->ScaleRows(penalty);
          block_J->SetBlock(0, 0, BTB.release());
        }
        else
        {
          block_J->SetBlock(0, 0, mfem::Add(
            1.0, static_cast<mfem::HypreParMatrix&>(block_J->GetBlock(0, 0)),
            contactPairs()[i].getContactOptions().penalty, *BTB
          ));
        }
      }
      else // enforcement == ContactEnforcement::LagrangeMultiplier
      {
        constraint_matrices(static_cast<int>(i), 0) = 
          static_cast<mfem::HypreParMatrix*>(&pair_J->GetBlock(1, 0));
      }
      if (
        pair_J->IsZeroBlock(0, 1) || 
        !dynamic_cast<mfem::TransposeOperator*>(&pair_J->GetBlock(0, 1))
      )
      {
        SLIC_ERROR_ROOT("Only symmetric constraint matrices are currently supported.");
      }
      delete &pair_J->GetBlock(0, 1);
      if (!pair_J->IsZeroBlock(1, 1))
      {
        SLIC_ERROR_ROOT("Only zero-valued (1, 1) Jacobian blocks are currently supported.");
      }
    }
  }
  if (numPressureTrueDofs() > 0)
  {
    block_J->SetBlock(1, 0, mfem::HypreParMatrixFromBlocks(constraint_matrices));
    block_J->SetBlock(0, 1, new mfem::TransposeOperator(block_J->GetBlock(1, 0)));
  }
#endif
  return block_J;
}

mfem::Array<int> ContactData::pressureTrueDofOffsets() const
{
  mfem::Array<int> dof_offsets(static_cast<int>(contactPairs().size()) + 1);
  for (size_t i{0}; i < contactPairs().size(); ++i)
  {
    dof_offsets[static_cast<int>(i+1)] = 
      dof_offsets[static_cast<int>(i)] + contactPairs()[i].numTruePressureDofs();
  }
  return dof_offsets;
}

void ContactData::setPressures([[maybe_unused]] const mfem::Vector& true_pressures) const
{
#ifdef SERAC_USE_TRIBOL
  auto dof_offsets = pressureTrueDofOffsets();
  for (size_t i{0}; i < contactPairs().size(); ++i)
  {
    if (contactPairs()[i].getContactOptions().enforcement == ContactEnforcement::LagrangeMultiplier)
    {
      mfem::Vector p_pair_true;
      p_pair_true.MakeRef(
        const_cast<mfem::Vector&>(true_pressures), 
        dof_offsets[static_cast<int>(i)],
        dof_offsets[static_cast<int>(i)+1] - dof_offsets[static_cast<int>(i)]
      );
      contactPairs()[i].pressure().ParFESpace()->GetRestrictionMatrix()->MultTranspose(
        p_pair_true,
        contactPairs()[i].pressure()
      );
    }
    else // enforcement == ContactEnforcement::Penalty
    {
      contactPairs()[i].pressure().Set(
        contactPairs()[i].getContactOptions().penalty, 
        contactPairs()[i].gaps()
      );
    }
  }
#endif
}

void ContactData::setDisplacements([[maybe_unused]] const mfem::Vector& true_displacement)
{
#ifdef SERAC_USE_TRIBOL
  reference_nodes_->ParFESpace()->GetProlongationMatrix()->Mult(true_displacement, current_coords_);
  current_coords_ += *reference_nodes_;
#endif
}

}  // namespace serac
