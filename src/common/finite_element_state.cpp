// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "common/serac_types.hpp"

namespace serac {

FiniteElementState::FiniteElementState(const int order, std::shared_ptr<mfem::ParMesh> pmesh, const FESOptions& options)
    : mesh_(pmesh),
      coll_(options.coll ? *options.coll : std::make_shared<mfem::H1_FECollection>(order, pmesh->Dimension())),
      space_(std::make_shared<mfem::ParFiniteElementSpace>(
          pmesh.get(), coll_.get(), options.space_dim ? *options.space_dim : pmesh->Dimension(), options.ordering)),
      gf_(std::make_shared<mfem::ParGridFunction>(space_.get())),
      true_vec_(std::make_shared<mfem::HypreParVector>(space_.get())),
      name_(options.name)
{
  *gf_       = 0.0;
  *true_vec_ = 0.0;
}

}  // namespace serac
