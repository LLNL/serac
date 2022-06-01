// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/state/finite_element_vector.hpp"

namespace serac {

FiniteElementVector::FiniteElementVector(mfem::ParMesh& mesh, FiniteElementVector::Options&& options)
    : mesh_(mesh),
      coll_(options.coll ? std::move(options.coll)
                         : std::make_unique<mfem::H1_FECollection>(options.order, mesh.Dimension())),
      space_(std::make_unique<mfem::ParFiniteElementSpace>(&mesh, coll_.get(), options.vector_dim, options.ordering)),
      name_(options.name),
      true_vec_(space_.get())
{
  true_vec_ = 0.0;
}

FiniteElementVector::FiniteElementVector(mfem::ParMesh& mesh, const mfem::ParFiniteElementSpace& space,
                                         const std::string& name)
    : mesh_(mesh),
      coll_(std::unique_ptr<mfem::FiniteElementCollection>(mfem::FiniteElementCollection::New(space.FEColl()->Name()))),
      space_(std::make_unique<mfem::ParFiniteElementSpace>(space, &mesh, coll_.get())),
      name_(name),
      true_vec_(space_.get())
{
  true_vec_ = 0.0;
}

FiniteElementVector::FiniteElementVector(const FiniteElementVector& input_vector)
    : FiniteElementVector(input_vector.mesh_.get(), *input_vector.space_, input_vector.name_)
{
  true_vec_ = input_vector.true_vec_;
}

FiniteElementVector::FiniteElementVector(FiniteElementVector&& input_vector)
    : mesh_(input_vector.mesh()),
      coll_(std::move(input_vector.coll_)),
      space_(std::move(input_vector.space_)),
      name_(std::move(input_vector.name_)),
      true_vec_(std::move(input_vector.true_vec_))
{
}

FiniteElementVector& FiniteElementVector::operator=(const double value)
{
  true_vec_ = value;
  return *this;
}

double avg(const FiniteElementVector& fe_vector)
{
  double global_sum;
  double local_sum = fe_vector.true_vec_.Sum();
  int    global_size;
  int    local_size = fe_vector.true_vec_.Size();
  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, fe_vector.comm());
  MPI_Allreduce(&local_size, &global_size, 1, MPI_INT, MPI_SUM, fe_vector.comm());
  return global_sum / global_size;
}

double max(const FiniteElementVector& fe_vector)
{
  double global_max;
  double local_max = fe_vector.true_vec_.Max();
  MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, fe_vector.comm());
  return global_max;
}

double min(const FiniteElementVector& fe_vector)
{
  double global_min;
  double local_min = fe_vector.true_vec_.Min();
  MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, fe_vector.comm());
  return global_min;
}

double norm(const FiniteElementVector& state, const double p)
{
  if (state.space().GetVDim() == 1) {
    mfem::ConstantCoefficient zero(0.0);
    return state.gridFunction().ComputeLpError(p, zero);
  } else {
    mfem::Vector zero(state.space().GetVDim());
    zero = 0.0;
    mfem::VectorConstantCoefficient zerovec(zero);
    return state.gridFunction().ComputeLpError(p, zerovec);
  }
}

}  // namespace serac
