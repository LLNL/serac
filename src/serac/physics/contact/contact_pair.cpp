// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/contact/contact_pair.hpp"

#ifdef SERAC_USE_TRIBOL

#include "axom/slic.hpp"

#include "serac/physics/contact/contact_config.hpp"

#include "tribol/interface/tribol.hpp"
#include "tribol/interface/mfem_tribol.hpp"

namespace serac {

ContactPair::ContactPair(int pair_id, const mfem::ParMesh& mesh,
                         const std::set<int>& bdry_attr_surf1,
                         const std::set<int>& bdry_attr_surf2,
                         const mfem::ParGridFunction& current_coords, ContactOptions contact_opts)
    : pair_id_{pair_id}, contact_opts_{contact_opts}, current_coords_{current_coords}
{
  tribol::registerMfemCouplingScheme(pair_id, 2 * pair_id, 2 * pair_id + 1, mesh, current_coords, bdry_attr_surf1,
                                     bdry_attr_surf2, tribol::SURFACE_TO_SURFACE, tribol::NO_SLIDING, getMethod(),
                                     tribol::FRICTIONLESS, tribol::LAGRANGE_MULTIPLIER);
  tribol::setLagrangeMultiplierOptions(pair_id, tribol::ImplicitEvalMode::MORTAR_RESIDUAL_JACOBIAN);
  
  // get true DOFs only associated with surface 1
  if (getContactOptions().type == ContactType::TiedSlide) {
    auto& pressure_space = *tribol::getMfemPressure(pair_id).ParFESpace();
    mfem::Array<int> dof_markers(pressure_space.GetVSize());
    dof_markers = 1;
    mfem::Array<int> surf2_markers(pressure_space.GetMesh()->attributes.Max());
    for (auto attr : bdry_attr_surf2) {
      surf2_markers[attr - 1] = 1;
    }
    for (int e{0}; e < pressure_space.GetNE(); ++e) {
      if (surf2_markers[pressure_space.GetAttribute(e) - 1]) {
        mfem::Array<int> vdofs;
        pressure_space.GetElementVDofs(e, vdofs);
        for (int d{0}; d < vdofs.Size(); ++d) {
          int k = vdofs[d];
          if (k < 0) { k = -1 - k; }
          dof_markers[k] = 0;
        }
      }
    }
    mfem::Array<int> tdof_markers(pressure_space.GetTrueVSize());
    pressure_space.GetRestrictionMatrix()->BooleanMult(dof_markers, tdof_markers);
    mfem::FiniteElementSpace::MarkerToList(tdof_markers, inactive_tdofs_);
  }
}

mfem::Vector ContactPair::contactForces() const
{
  mfem::Vector f(current_coords_.ParFESpace()->GetVSize());
  f = 0.0;
  tribol::getMfemResponse(getPairId(), f);
  return f;
}

mfem::Vector ContactPair::gaps() const
{
  mfem::Vector g;
  tribol::getMfemGap(getPairId(), g);
  return g;
}

mfem::ParGridFunction& ContactPair::pressure() const
{
  return tribol::getMfemPressure(getPairId());
}

int ContactPair::numPressureTrueDofs() const
{
  return getContactOptions().enforcement == ContactEnforcement::LagrangeMultiplier
             ? tribol::getMfemPressure(getPairId()).ParFESpace()->GetTrueVSize()
             : 0;
}

tribol::ContactMethod ContactPair::getMethod() const
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

const mfem::Array<int>& ContactPair::inactiveTrueDofs() const
{
  if (getContactOptions().type == ContactType::Frictionless) {
    auto& p = pressure();
    auto g = gaps();
    mfem::Vector p_true(p.ParFESpace()->GetTrueVSize());
    p.ParFESpace()->GetRestrictionOperator()->Mult(p, p_true);
    mfem::Vector g_true(p.ParFESpace()->GetTrueVSize());
    p.ParFESpace()->GetRestrictionOperator()->Mult(g, g_true);
    std::vector<int> inactive_tdofs_vector;
    inactive_tdofs_vector.reserve(static_cast<size_t>(p_true.Size()));
    for (int d{0}; d < p_true.Size(); ++d) {
      if (p_true[d] >= 0.0 && g_true[d] >= -1.0e-14) {
        inactive_tdofs_vector.push_back(d);
      }
    }
    inactive_tdofs_ = mfem::Array<int>(static_cast<int>(inactive_tdofs_vector.size()));
    std::copy(inactive_tdofs_vector.begin(), inactive_tdofs_vector.end(), inactive_tdofs_.begin());
  }
  return inactive_tdofs_;
}

}  // namespace serac

#endif
