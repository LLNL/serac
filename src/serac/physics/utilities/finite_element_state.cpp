// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
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
      space_(&mesh, coll_.get(), options.vector_dim, options.ordering),
      gf_(std::make_unique<mfem::ParGridFunction>(&space_)),
      true_vec_(&space_),
      name_(options.name)
{
  *gf_      = 0.0;
  true_vec_ = 0.0;
}

}  // namespace serac
