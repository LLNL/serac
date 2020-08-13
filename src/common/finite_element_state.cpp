// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "common/finite_element_state.hpp"

namespace serac {

FiniteElementState::FiniteElementState(const int order, std::shared_ptr<mfem::ParMesh> pmesh, FESOptions&& options)
    : mesh_(pmesh),
      coll_(options.coll ? std::move(*options.coll)
                         : std::make_unique<mfem::H1_FECollection>(order, pmesh->Dimension())),
      space_(pmesh.get(), coll_.get(), options.space_dim ? *options.space_dim : pmesh->Dimension(), options.ordering),
      gf_(std::make_unique<mfem::ParGridFunction>(&space_)),
      true_vec_(&space_),
      name_(options.name)
{
  *gf_      = 0.0;
  true_vec_ = 0.0;
}

}  // namespace serac
