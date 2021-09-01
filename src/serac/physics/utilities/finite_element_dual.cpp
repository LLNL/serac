// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/utilities/finite_element_dual.hpp"
#include "serac/infrastructure/logger.hpp"

namespace serac {

FiniteElementDual::FiniteElementDual(mfem::ParMesh& mesh, FiniteElementDual::Options&& options)
    : mesh_(mesh),
      coll_(options.coll ? std::move(options.coll)
                         : std::make_unique<mfem::H1_FECollection>(options.order, mesh.Dimension())),
      space_(std::make_unique<mfem::ParFiniteElementSpace>(&mesh, coll_.get(), options.vector_dim, options.ordering)),
      local_vec_(space_->GetVSize()),
      true_vec_(space_.get()),
      name_(options.name)
{
  true_vec_  = 0.0;
  local_vec_ = 0.0;
}

FiniteElementDual::FiniteElementDual(mfem::ParMesh& mesh, mfem::ParFiniteElementSpace& space, const std::string& name)
    : mesh_(mesh),
      coll_(std::unique_ptr<mfem::FiniteElementCollection>(mfem::FiniteElementCollection::New(space.FEColl()->Name()))),
      space_(std::make_unique<mfem::ParFiniteElementSpace>(space, &mesh, coll_.get())),
      local_vec_(space_->GetVSize()),
      true_vec_(space_.get()),
      name_(name)
{
  true_vec_  = 0.0;
  local_vec_ = 0.0;
}

void FiniteElementDual::initializeTrueVec()
{
  auto* R = space_->GetRestrictionMatrix();
  if (!R || mfem::IsIdentityProlongation(space_->GetProlongationMatrix())) {
    // The true dofs and the local dofs are the same
    static_cast<mfem::Vector>(true_vec_) = local_vec_;
  } else {
    R->Mult(local_vec_, true_vec_);
  }
}

void FiniteElementDual::distributeSharedDofs()
{
  auto* P = space_->GetProlongationMatrix();
  if (!P) {
    // The true dofs and the local dofs are the same
    local_vec_ = true_vec_;
  } else {
    P->Mult(true_vec_, local_vec_);
  }
}

FiniteElementDual& FiniteElementDual::operator=(const double value)
{
  true_vec_ = value;
  distributeSharedDofs();
  return *this;
}

}  // namespace serac
