// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/utilities/finite_element_state.hpp"

namespace serac {

FiniteElementState::FiniteElementState(mfem::ParMesh& mesh, FiniteElementState::Options&& options)
    : mesh_(mesh),
      coll_(options.coll ? std::move(options.coll)
                         : std::make_unique<mfem::H1_FECollection>(options.order, mesh.Dimension())),
      space_(
          std::make_unique<mfem::ParFiniteElementSpace>(&mesh, &retrieve(coll_), options.vector_dim, options.ordering)),
      // When left unallocated, the allocation can happen inside the datastore
      // Use a raw pointer here when unallocated, lifetime will be managed by the DataCollection
      gf_(options.alloc_gf
              ? MaybeOwningPointer<mfem::ParGridFunction>{std::make_unique<mfem::ParGridFunction>(&retrieve(space_))}
              : MaybeOwningPointer<mfem::ParGridFunction>{new mfem::ParGridFunction(&retrieve(space_),
                                                                                    static_cast<double*>(nullptr))}),
      true_vec_(&retrieve(space_)),
      name_(options.name)
{
  true_vec_ = 0.0;
}

FiniteElementState::FiniteElementState(mfem::ParMesh& mesh, mfem::ParGridFunction& gf, const std::string& name)
    : mesh_(mesh), space_(gf.ParFESpace()), gf_(&gf), true_vec_(&retrieve(space_)), name_(name)
{
  coll_     = retrieve(space_).FEColl();
  true_vec_ = 0.0;
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
