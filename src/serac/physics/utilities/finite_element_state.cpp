// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/utilities/finite_element_state.hpp"

#include "mpi.h"

#include "serac/infrastructure/initialize.hpp"

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

double avg(const FiniteElementState& state)
{
  double global_sum, local_sum   = state.trueVec().Sum();
  int    global_size, local_size = state.trueVec().Size();
  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, state.comm());
  MPI_Allreduce(&local_size, &global_size, 1, MPI_INT, MPI_SUM, state.comm());
  return global_sum / global_size;
}

double max(const FiniteElementState& state)
{
  double global_max, local_max = state.trueVec().Max();
  MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, state.comm());
  return global_max;
}

double min(const FiniteElementState& state)
{
  double global_min, local_min = state.trueVec().Min();
  MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, state.comm());
  return global_min;
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
