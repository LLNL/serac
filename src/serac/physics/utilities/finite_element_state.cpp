// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/utilities/finite_element_state.hpp"

namespace serac {

namespace detail {

/**
 * @brief Helper function for creating a GridFunction in both restart and not-restart scenarios
 * @param[in] space The FESpace to construct the GridFunction with
 * @param[in] alloc_gf Whether to allocate the GridFunction - if this is a non-restart run, we delay the allocation
 * so it can be taken care of inside MFEMSidreDataCollection
 */
MaybeOwningPointer<mfem::ParGridFunction> initialGridFunc(mfem::ParFiniteElementSpace* space, const bool alloc_gf)
{
  if (alloc_gf) {
    return std::make_unique<mfem::ParGridFunction>(space);
  } else {
    return new mfem::ParGridFunction(space, static_cast<double*>(nullptr));
  }
}

}  // namespace detail

FiniteElementState::FiniteElementState(mfem::ParMesh& mesh, FiniteElementState::Options&& options)
    : mesh_(mesh),
      coll_(options.coll ? std::move(options.coll)
                         : std::make_unique<mfem::H1_FECollection>(options.order, mesh.Dimension())),
      space_(std::make_unique<mfem::ParFiniteElementSpace>(&mesh, &detail::retrieve(coll_), options.vector_dim,
                                                           options.ordering)),
      // When left unallocated, the allocation can happen inside the datastore
      // Use a raw pointer here when unallocated, lifetime will be managed by the DataCollection
      gf_(detail::initialGridFunc(&detail::retrieve(space_), options.alloc_gf)),
      true_vec_(&detail::retrieve(space_)),
      name_(options.name)
{
  // Add check to make sure order and space match, if a collection is provided the order is not used
  // It seems mfem allows each element to be of a different order, but most input to FiniteElementState probably has the
  // same order; thus we'll sue a soft warning here.
  SLIC_WARNING_IF(options.order != (&retrieve(space_))->GetOrder(0),
                  "The order specified in options may not match the space");
  true_vec_ = 0.0;
}

FiniteElementState::FiniteElementState(mfem::ParMesh& mesh, mfem::ParGridFunction& gf, const std::string& name)
    : mesh_(mesh), space_(gf.ParFESpace()), gf_(&gf), true_vec_(&detail::retrieve(space_)), name_(name)
{
  coll_     = detail::retrieve(space_).FEColl();
  true_vec_ = 0.0;
}

FiniteElementState::FiniteElementState(mfem::ParMesh& mesh, FiniteElementState& fe_state, const std::string& name)
    : FiniteElementState(mesh, fe_state.gridFunc(), name)
{
}

FiniteElementState& FiniteElementState::operator=(const double value)
{
  true_vec_ = value;
  distributeSharedDofs();
  return *this;
}

double norm(const FiniteElementState& state, const double p)
{
  if (state.space().GetVDim() == 1) {
    mfem::ConstantCoefficient zero(0.0);
    return state.gridFunc().ComputeLpError(p, zero);
  } else {
    mfem::Vector zero(state.space().GetVDim());
    zero = 0.0;
    mfem::VectorConstantCoefficient zerovec(zero);
    return state.gridFunc().ComputeLpError(p, zerovec);
  }
}

}  // namespace serac
