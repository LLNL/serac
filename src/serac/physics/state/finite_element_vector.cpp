// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/state/finite_element_vector.hpp"
#include "serac/infrastructure/logger.hpp"

namespace serac {

hypre_ParVector* OwnsData::createVector(mfem::ParFiniteElementSpace& space, [[maybe_unused]] double* data) const
{
  mfem::HypreParVector new_vector(&space);
  // Initialize the vector to zero
  new_vector = 0.0;
  return new_vector.StealParVector();
}

void OwnsData::createCopy(mfem::Vector& to, const mfem::Vector& from) const
{
  to = from;
}

hypre_ParVector* DataView::createVector(mfem::ParFiniteElementSpace& space, double* data) const
{
  SLIC_ERROR_ROOT_IF(!data, "data is null. A view-type FiniteElementVector must point to data.");
  mfem::HypreParVector new_vector(space.GetComm(), space.GlobalTrueVSize(), data, space.GetTrueDofOffsets());
  return new_vector.StealParVector();
}

void DataView::createCopy([[maybe_unused]] mfem::Vector& to, [[maybe_unused]] const mfem::Vector& from) const
{
  SLIC_ERROR_ROOT("Copy constructor is not supported for view-type FiniteElementVectors.");
  //to.SetData(from.GetData());
}

FiniteElementVector::FiniteElementVector(mfem::ParMesh& mesh, FiniteElementVector::Options&& options)
    : mesh_(mesh), name_(options.name), data_relationship_(new class OwnsData)
{
  const int  dim      = mesh.Dimension();
  const auto ordering = mfem::Ordering::byNODES;

  switch (options.element_type) {
    case ElementType::H1:
      coll_ = std::make_unique<mfem::H1_FECollection>(options.order, dim);
      break;
    case ElementType::HCURL:
      coll_ = std::make_unique<mfem::ND_FECollection>(options.order, dim);
      SLIC_WARNING_ROOT_IF(options.vector_dim != 1,
                           axom::fmt::format("Vector dim >1 requested for an HCURL basis function."));
      break;
    case ElementType::HDIV:
      coll_ = std::make_unique<mfem::RT_FECollection>(options.order, dim);
      SLIC_WARNING_ROOT_IF(options.vector_dim != 1,
                           axom::fmt::format("Vector dim >1 requested for an HDIV basis function."));
      break;
    case ElementType::L2:
      // Note that we use Gauss-Lobatto basis functions as this is what serac::Functional uses for finite element
      // integrals
      coll_ = std::make_unique<mfem::L2_FECollection>(options.order, dim, mfem::BasisType::GaussLobatto);
      break;
    default:
      SLIC_ERROR_ROOT(axom::fmt::format("Finite element vector requested for unavailable basis type."));
      break;
  }

  space_ = std::make_unique<mfem::ParFiniteElementSpace>(&mesh, coll_.get(), options.vector_dim, ordering);

  // Construct a hypre par vector based on the new finite element space
  // Move the data from this new hypre vector into this object without doubly allocating the data
  WrapHypreParVector(data_relationship_->createVector(*space_));
}

FiniteElementVector::FiniteElementVector(const mfem::ParFiniteElementSpace& space, const std::string& name,
                                         std::unique_ptr<DataRelationship> data_relationship, double* view_data)
    : mesh_(*space.GetParMesh()),
      coll_(std::unique_ptr<mfem::FiniteElementCollection>(mfem::FiniteElementCollection::New(space.FEColl()->Name()))),
      space_(std::make_unique<mfem::ParFiniteElementSpace>(space, &mesh_.get(), coll_.get())),
      name_(name),
      data_relationship_(std::move(data_relationship))
{
  SLIC_ERROR_ROOT_IF(space.GetOrdering() == mfem::Ordering::byVDIM,
                     "Serac only operates on finite element spaces ordered by nodes");

  // Construct a hypre par vector based on the new finite element space
  // Move the data from this new hypre vector into this object without doubly allocating the data
  WrapHypreParVector(data_relationship_->createVector(*space_, view_data));
}

FiniteElementVector::FiniteElementVector(FiniteElementVector&& input_vector)
    : mesh_(input_vector.mesh()),
      coll_(std::move(input_vector.coll_)),
      space_(std::move(input_vector.space_)),
      name_(std::move(input_vector.name_)),
      data_relationship_(std::move(input_vector.data_relationship_))
{
  // Grab the allocated data from the input argument for the underlying Hypre vector
  auto* parallel_vec = input_vector.StealParVector();
  WrapHypreParVector(parallel_vec);
}

FiniteElementVector& FiniteElementVector::operator=(const mfem::HypreParVector& rhs)
{
  SLIC_ERROR_IF(Size() != rhs.Size(),
                axom::fmt::format("Finite element vector of size '{}' assigned to a HypreParVector of size '{}'",
                                  Size(), rhs.Size()));

  HypreParVector::operator=(rhs);
  return *this;
}

FiniteElementVector& FiniteElementVector::operator=(const FiniteElementVector& rhs)
{
  SLIC_ERROR_IF(Size() != rhs.Size(),
                axom::fmt::format("Finite element vector of size '{}' assigned to a HypreParVector of size '{}'",
                                  Size(), rhs.Size()));

  HypreParVector::operator=(rhs);

  return *this;
}

FiniteElementVector& FiniteElementVector::operator=(FiniteElementVector&& rhs)
{
  mesh_  = rhs.mesh_;
  coll_  = std::move(rhs.coll_);
  space_ = std::move(rhs.space_);
  name_  = rhs.name_;
  data_relationship_ = std::move(rhs.data_relationship_);

  auto* parallel_vec = rhs.StealParVector();
  WrapHypreParVector(parallel_vec);

  return *this;
}

FiniteElementVector& FiniteElementVector::operator=(const double value)
{
  HypreParVector::operator=(value);
  return *this;
}

double avg(const FiniteElementVector& fe_vector)
{
  double global_sum;
  double local_sum = fe_vector.Sum();
  int    global_size;
  int    local_size = fe_vector.Size();
  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, fe_vector.comm());
  MPI_Allreduce(&local_size, &global_size, 1, MPI_INT, MPI_SUM, fe_vector.comm());
  return global_sum / global_size;
}

double max(const FiniteElementVector& fe_vector)
{
  double global_max;
  double local_max = fe_vector.Max();
  MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, fe_vector.comm());
  return global_max;
}

double min(const FiniteElementVector& fe_vector)
{
  double global_min;
  double local_min = fe_vector.Min();
  MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, fe_vector.comm());
  return global_min;
}

}  // namespace serac
