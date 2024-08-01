// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/contact/contact_interaction.hpp"

#ifdef SERAC_USE_TRIBOL

#include "axom/slic.hpp"

#include "serac/physics/contact/contact_config.hpp"

#include "tribol/interface/tribol.hpp"
#include "tribol/interface/mfem_tribol.hpp"

namespace serac {

static void mark_dofs(const mfem::Array<int>& dofs, mfem::Array<int>& mark_array)
{
  for (int i = 0; i < dofs.Size(); i++) {
    int k = dofs[i];
    if (k < 0) {
      k = -1 - k;
    }
    mark_array[k] = 0;
  }
}

ContactInteraction::ContactInteraction(int interaction_id, const mfem::ParMesh& mesh,
                                       const std::set<int>& bdry_attr_surf1, const std::set<int>& bdry_attr_surf2,
                                       const mfem::ParGridFunction& current_coords, ContactOptions contact_opts)
    : interaction_id_{interaction_id}, contact_opts_{contact_opts}
{
  int mesh1_id = 2 * interaction_id;      // unique id for the first Tribol mesh
  int mesh2_id = 2 * interaction_id + 1;  // unique id for the second Tribol mesh
  tribol::registerMfemCouplingScheme(interaction_id, mesh1_id, mesh2_id, mesh, current_coords, bdry_attr_surf1,
                                     bdry_attr_surf2, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE, getMethod(),
                                     tribol::FRICTIONLESS, tribol::LAGRANGE_MULTIPLIER);
  tribol::setLagrangeMultiplierOptions(interaction_id, tribol::ImplicitEvalMode::MORTAR_RESIDUAL_JACOBIAN);

  pressure_ = std::make_unique<FiniteElementState>(pressureSpace());
  gaps_     = std::make_unique<FiniteElementDual>(pressureSpace());
  forces_   = std::make_unique<FiniteElementDual>(*current_coords.ParFESpace());
  inactive_tdofs_.Reserve(pressure_->Size());
  SLIC_INFO("pressureSpace().GetTrueVSize() = " << pressureSpace().GetTrueVSize());

  // get true DOFs only associated with surface 1 (i.e. surface 1 \ surface 2)
  if (getContactOptions().type == ContactType::TiedNormal) {
    // this block essentially returns the complement of GetEssentialTrueDofsFromElementAttribute(surface 2) (def'd in
    // boundary_condition_helper)
    auto&            pressure_space = *tribol::getMfemPressure(interaction_id).ParFESpace();
    mfem::Array<int> dof_markers(pressure_space.GetVSize());
    dof_markers = -1;
    mfem::Array<int> surf2_markers(pressure_space.GetMesh()->attributes.Max());
    for (auto attr : bdry_attr_surf2) {
      surf2_markers[attr - 1] = 1;
    }
    for (int e{0}; e < pressure_space.GetNE(); ++e) {
      if (surf2_markers[pressure_space.GetAttribute(e) - 1]) {
        mfem::Array<int> vdofs;
        pressure_space.GetElementVDofs(e, vdofs);
        mark_dofs(vdofs, dof_markers);
      }
    }
    mfem::Array<int> tdof_markers(pressure_space.GetTrueVSize());
    pressure_space.GetRestrictionMatrix()->BooleanMult(dof_markers, tdof_markers);
    mfem::FiniteElementSpace::MarkerToList(tdof_markers, inactive_tdofs_);
  }
}

void ContactInteraction::update()
{
  updateGaps();
  updateForces();
  updatePressure();
  updateInactiveDofs();
  serac::logger::flush();
}

void ContactInteraction::updateForces()
{
  *forces_    = 0.0;
  auto& f_loc = forces_->linearForm();
  tribol::getMfemResponse(getInteractionId(), f_loc);
  // Equivalent to:ß
  // current_coords_.ParFESpace()->GetProlongationMatrix()->MultTranspose(f_loc, f);
  forces_->setFromLinearForm(f_loc);
}

const FiniteElementDual& ContactInteraction::forces() const { return *forces_; }

void ContactInteraction::updatePressure()
{
  // Equivalent to:
  // pressureSpace().GetRestrictionMatrix()->Mult();
  pressure_->setFromGridFunction(tribol::getMfemPressure(getInteractionId()));
}

const FiniteElementState& ContactInteraction::pressure() const { return *pressure_; }

void ContactInteraction::updateGaps()
{
  mfem::Vector g_loc;
  tribol::getMfemGap(getInteractionId(), g_loc);
  // Equivalent to:
  // pressureSpace().GetProlongationMatrix()->MultTranspose(g_loc, g);
  pressureSpace().GetProlongationMatrix()->MultTranspose(g_loc, *gaps_);
  // gaps_->setFromLinearForm(g_loc);
  SLIC_INFO("Minimum gap: (tdofs): " << gaps_->Min() << "; (ldofs): " << g_loc.Min());
  SLIC_INFO("Maximum gap: (tdofs): " << gaps_->Max() << "; (ldofs): " << g_loc.Max());
}

const FiniteElementDual& ContactInteraction::gaps() const { return *gaps_; }

std::unique_ptr<mfem::BlockOperator> ContactInteraction::jacobian() const
{
  return tribol::getMfemBlockJacobian(getInteractionId());
}

int ContactInteraction::numPressureDofs() const
{
  return getContactOptions().enforcement == ContactEnforcement::LagrangeMultiplier
             ? tribol::getMfemPressure(getInteractionId()).ParFESpace()->GetTrueVSize()
             : 0;
}

mfem::ParFiniteElementSpace& ContactInteraction::pressureSpace() const
{
  return *tribol::getMfemPressure(getInteractionId()).ParFESpace();
}

void ContactInteraction::setPressure(const FiniteElementState& pressure)
{
  *pressure_ = pressure;
  pressure_->fillGridFunction(tribol::getMfemPressure(getInteractionId()));
}

void ContactInteraction::updateInactiveDofs()
{
  if (getContactOptions().type == ContactType::Frictionless) {
    inactive_tdofs_.SetSize(0);
    auto& p = pressure();
    auto& g = gaps();
    for (int d{0}; d < p.Size(); ++d) {
      if (p[d] >= 0.0 && g[d] >= -1.0e-14) {
        inactive_tdofs_.Append(d);
      }
    }
  }
}

const mfem::Array<int>& ContactInteraction::inactiveDofs() const { return inactive_tdofs_; }

tribol::ContactMethod ContactInteraction::getMethod() const
{
  switch (contact_opts_.method) {
    case ContactMethod::SingleMortar:
      return tribol::SINGLE_MORTAR;
      break;
    default:
      SLIC_ERROR_ROOT("Unsupported contact method.");
      // return something so we don't get an error
      return tribol::SINGLE_MORTAR;
  }
}

}  // namespace serac

#endif
