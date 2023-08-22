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

}  // namespace serac

#endif
