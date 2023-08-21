// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/contact/contact_pair.hpp"

#include "axom/slic.hpp"

#include "serac/physics/contact/contact_config.hpp"

#ifdef SERAC_USE_TRIBOL
#include "tribol/interface/tribol.hpp"
#include "tribol/interface/mfem_tribol.hpp"
#endif

namespace serac {

ContactPair::ContactPair(
  int pair_id,
  [[maybe_unused]] const mfem::ParMesh& mesh,
  [[maybe_unused]] const std::set<int>& bdry_attr_surf1,
  [[maybe_unused]] const std::set<int>& bdry_attr_surf2,
  const mfem::ParGridFunction& current_coords,
  ContactOptions contact_opts
)
: pair_id_ { pair_id },
  contact_opts_ { contact_opts },
  current_coords_ { current_coords }
{
#ifdef SERAC_USE_TRIBOL
  tribol::registerMfemCouplingScheme(
    pair_id, 
    2*pair_id,
    2*pair_id + 1,
    mesh,
    current_coords,
    bdry_attr_surf1,
    bdry_attr_surf2, 
    tribol::SURFACE_TO_SURFACE,
    tribol::NO_SLIDING,
    getMethod(),
    tribol::FRICTIONLESS,
    tribol::LAGRANGE_MULTIPLIER
  );

  tribol::setLagrangeMultiplierOptions(
    pair_id, 
    tribol::ImplicitEvalMode::MORTAR_RESIDUAL_JACOBIAN
  );
#endif
}

mfem::Vector ContactPair::contactForces() const
{
  mfem::Vector f(current_coords_.ParFESpace()->GetVSize());
  f = 0.0;
#ifdef SERAC_USE_TRIBOL
  tribol::getMfemResponse(getPairId(), f);
#endif
  return f;
}

mfem::Vector ContactPair::gaps() const
{
  mfem::Vector g;
#ifdef SERAC_USE_TRIBOL
  tribol::getMfemGap(getPairId(), g);
#endif
  return g;
}

mfem::ParGridFunction& ContactPair::pressure() const
{
#ifdef SERAC_USE_TRIBOL
  return tribol::getMfemPressure(getPairId());
#else
  SLIC_ERROR_ROOT("Serac built without Tribol.");
  // just return some grid function lvalue to quiet compiler
  return const_cast<mfem::ParGridFunction&>(current_coords_);
#endif
}

int ContactPair::numTruePressureDofs() const
{
#ifdef SERAC_USE_TRIBOL
  return getContactOptions().enforcement == ContactEnforcement::LagrangeMultiplier ?
    tribol::getMfemPressure(getPairId()).ParFESpace()->GetTrueVSize() :
    0;
#else
  return 0;
#endif
}

#ifdef SERAC_USE_TRIBOL
tribol::ContactMethod ContactPair::getMethod() const
{
  switch (contact_opts_.method)
  {
    case ContactMethod::SingleMortar:
      return tribol::SINGLE_MORTAR;
      break;
    default:
      SLIC_ERROR_ROOT("Unsupported contact method.");
      // return something so we don't get an error
      return tribol::SINGLE_MORTAR;
  }
}
#endif

}  // namespace serac
