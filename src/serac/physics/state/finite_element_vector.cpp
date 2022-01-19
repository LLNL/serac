// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/state/finite_element_vector.hpp"

namespace serac {

namespace detail {

/**
 * @brief Helper function for creating a GridFunction in both sidre and not-sidre scenarios
 * @param[in] space The FESpace to construct the GridFunction with
 * @param[in] managed_by_sidre Whether the grid function is managed by sidre. If it is, the memory is managed by a raw
 * pointer instead of a unique ptr within the MFEM sidre data collection.
 */
MaybeOwningPointer<mfem::ParGridFunction> initialGridFunc(mfem::ParFiniteElementSpace* space,
                                                          const bool                   managed_by_sidre)
{
  if (!managed_by_sidre) {
    return std::make_unique<mfem::ParGridFunction>(space);
  } else {
    return new mfem::ParGridFunction(space, static_cast<double*>(nullptr));
  }
}

}  // namespace detail

FiniteElementVector::FiniteElementVector(mfem::ParMesh& mesh, FiniteElementVector::Options&& options)
    : mesh_(mesh),
      coll_(options.coll ? std::move(options.coll)
                         : std::make_unique<mfem::H1_FECollection>(options.order, mesh.Dimension())),
      space_(std::make_unique<mfem::ParFiniteElementSpace>(&mesh, &detail::retrieve(coll_), options.vector_dim,
                                                           options.ordering)),
      gf_(detail::initialGridFunc(&detail::retrieve(space_), options.managed_by_sidre)),
      true_vec_(&detail::retrieve(space_)),
      name_(options.name)
{
  true_vec_ = 0.0;
}

FiniteElementVector::FiniteElementVector(mfem::ParMesh& mesh, mfem::ParFiniteElementSpace& space,
                                         const std::string& name)
    : mesh_(mesh),
      coll_(std::unique_ptr<mfem::FiniteElementCollection>(mfem::FiniteElementCollection::New(space.FEColl()->Name()))),
      space_(std::make_unique<mfem::ParFiniteElementSpace>(space, &mesh, &detail::retrieve(coll_))),
      gf_(detail::initialGridFunc(&detail::retrieve(space_), false)),
      true_vec_(&detail::retrieve(space_)),
      name_(name)
{
  true_vec_ = 0.0;
}

FiniteElementVector::FiniteElementVector(mfem::ParMesh& mesh, mfem::ParGridFunction& gf, const std::string& name)
    : mesh_(mesh), space_(gf.ParFESpace()), gf_(&gf), true_vec_(&detail::retrieve(space_)), name_(name)
{
  coll_     = detail::retrieve(space_).FEColl();
  true_vec_ = 0.0;
}

FiniteElementVector::FiniteElementVector(FiniteElementVector&& input_vector)
    : mesh_(input_vector.mesh()),
      coll_(std::move(input_vector.coll_)),
      space_(std::move(input_vector.space_)),
      gf_(std::move(input_vector.gf_)),
      name_(std::move(input_vector.name_))
{
  // HypreParVec doesn't have a move constructor, so it must be implemented
  auto* parallel_vec = input_vector.true_vec_.StealParVector();
  true_vec_.WrapHypreParVector(parallel_vec);
}

FiniteElementVector& FiniteElementVector::operator=(const double value)
{
  true_vec_ = value;
  distributeSharedDofs();
  return *this;
}

double avg(const FiniteElementVector& fe_vector)
{
  double global_sum;
  double local_sum = fe_vector.trueVec().Sum();
  int    global_size;
  int    local_size = fe_vector.trueVec().Size();
  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, fe_vector.comm());
  MPI_Allreduce(&local_size, &global_size, 1, MPI_INT, MPI_SUM, fe_vector.comm());
  return global_sum / global_size;
}

double max(const FiniteElementVector& fe_vector)
{
  double global_max;
  double local_max = fe_vector.trueVec().Max();
  MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, fe_vector.comm());
  return global_max;
}

double min(const FiniteElementVector& fe_vector)
{
  double global_min;
  double local_min = fe_vector.trueVec().Min();
  MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, fe_vector.comm());
  return global_min;
}

}  // namespace serac
