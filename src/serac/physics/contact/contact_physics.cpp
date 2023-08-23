#include "serac/physics/contact/contact_physics.hpp"

#include "serac/infrastructure/logger.hpp"
#include "serac/serac_config.hpp"

#ifdef SERAC_USE_TRIBOL
#include "tribol/interface/tribol.hpp"

#endif

namespace serac {

#ifdef SERAC_USE_TRIBOL
ContactPhysics::ContactPhysics(mfem::ParMesh& mesh, serac::FiniteElementState& displacement,
                               mfem::ParGridFunction&                            reference_nodes,
                               std::vector<std::shared_ptr<serac::ContactData>>& contact_data)
    : mesh_(mesh),
      contact_bc_(contact_data),
      reference_nodes_(reference_nodes),
      gap_(mesh, serac::FiniteElementState::Options{.name = "contact_gap"}),
      pres_(mesh, serac::FiniteElementState::Options{.name = "contact_pressure"}),
      force_(mesh, serac::FiniteElementState::Options{.vector_dim = mesh.Dimension(), .name = "contact_forces"}),
      displacement_(displacement)
{
  int scalar_size = displacement_.space().GetNDofs();

  // The nodal postion vectors
  x_.SetSize(scalar_size);
  y_.SetSize(scalar_size);
  z_.SetSize(scalar_size);

  // The primary side force vectors
  p_fx_.SetSize(scalar_size);
  p_fy_.SetSize(scalar_size);
  p_fz_.SetSize(scalar_size);

  // The secondary side force vectors
  s_fx_.SetSize(scalar_size);
  s_fy_.SetSize(scalar_size);
  s_fz_.SetSize(scalar_size);

  p_fx_ = 0.0;
  p_fy_ = 0.0;
  p_fz_ = 0.0;

  s_fx_ = 0.0;
  s_fy_ = 0.0;
  s_fz_ = 0.0;

  // Set the element type
  mfem::Element::Type element_type = mesh_.GetElementType(0);

  switch (element_type) {
    case mfem::Element::POINT:
      elem_type_ = tribol::NODE;
      break;
    case mfem::Element::SEGMENT:
      elem_type_ = tribol::EDGE;
      break;
    case mfem::Element::QUADRILATERAL:
      elem_type_ = tribol::FACE;
      break;
    case mfem::Element::HEXAHEDRON:
      elem_type_ = tribol::CELL;
      break;

    case mfem::Element::TRIANGLE:
    case mfem::Element::TETRAHEDRON:
      SLIC_ERROR_ROOT("Unsupported element type!");
      break;

    default:
      SLIC_ERROR_ROOT("Unknown element type!");
      break;
  }

  mfem::Element::Type boundary_type = mesh_.GetBdrElementType(0);
  switch (boundary_type) {
    case mfem::Element::POINT:
      bdr_type_ = tribol::NODE;
      break;
    case mfem::Element::SEGMENT:
      bdr_type_ = tribol::EDGE;
      break;
    case mfem::Element::QUADRILATERAL:
      bdr_type_ = tribol::FACE;
      break;
    case mfem::Element::HEXAHEDRON:
      bdr_type_ = tribol::CELL;
      break;

    case mfem::Element::TRIANGLE:
    case mfem::Element::TETRAHEDRON:
      SLIC_ERROR_ROOT("Unsupported element type!");
      break;

    default:
      SLIC_ERROR_ROOT("Unknown element type!");
      break;
  }

  // Generate the mesh connectivites
  mfem::Array<int> total_bdr_vtx, bdr_attr;

  num_primary_faces_   = 0;
  num_secondary_faces_ = 0;

  num_verts_ = mfem::Geometry::NumVerts[boundary_type];

  // Fill the connectivity and node lists
  for (auto& contact_bc : contact_bc_) {
    // Get the boundary element attributes and vertices
    mesh_.GetBdrElementData(boundary_type, total_bdr_vtx, bdr_attr);

    // Loop over all boundary elements
    for (int i = 0; i < bdr_attr.Size(); ++i) {
      // If this is a primary side boundary element
      if (contact_bc->primary_markers[bdr_attr[i] - 1] != 0) {
        // Increment the counter
        num_primary_faces_++;

        // Loop over the vertices
        for (int v = 0; v < num_verts_; ++v) {
          // Add the node to the connectivity array
          p_conn_.push_back(total_bdr_vtx[i * num_verts_ + v]);

          // Add the node to the primary side set
          p_nodes_.insert(total_bdr_vtx[i * num_verts_ + v]);
        }
      }

      // Do the same with the secondary boundary elements
      if (contact_bc->secondary_markers[bdr_attr[i] - 1] != 0) {
        num_secondary_faces_++;
        for (int v = 0; v < num_verts_; ++v) {
          s_conn_.push_back(total_bdr_vtx[i * num_verts_ + v]);
          s_nodes_.insert(total_bdr_vtx[i * num_verts_ + v]);
        }
      }
    }
  }

  // initialize tribol
  tribol::initialize(mesh_.Dimension(), mesh_.GetComm());

  // Update the tribol mesh data
  displacement_ = 0.0;
  updatePosition();

  // Set the constraint matrix to empty
  full_constraint_matrix_ = nullptr;

  // set up the reduced contraint matrix
  constraint_matrix_ = nullptr;

  // Set the constraint rhs to the number of secondary nodes
  // Note: this assumes all secondary nodes are in contact
  constraint_rhs_.SetSize(static_cast<int>(s_nodes_.size()));
}

#else
ContactPhysics::ContactPhysics(mfem::ParMesh& mesh, serac::FiniteElementState&, mfem::ParGridFunction& reference_nodes,
                               std::vector<std::shared_ptr<serac::ContactData>>& contact_data)
    : mesh_(mesh),
      contact_bc_(contact_data),
      reference_nodes_(reference_nodes),
      gap_(mesh, serac::FiniteElementState::Options{.name = "contact_gap"}),
      pres_(mesh, serac::FiniteElementState::Options{.name = "contact_pressure"}),
      force_(mesh, serac::FiniteElementState::Options{.name = "contact_forces"})
{
  SLIC_ERROR_ROOT("Tribol not enabled in this build!");
}
#endif

// Update the tribol position data structures
void ContactPhysics::updatePosition()
{
#ifdef SERAC_USE_TRIBOL
  // Get the scalar number of degrees of freedom in the mesh
  int scalar_size = displacement_.space().GetNDofs();

  auto& displacement_gf = displacement_.gridFunction();

  // If ordering is by nodes, it is stacked x1, x2, x3..., y1, y2, y3, ..., z1, z2, z3...
  if (displacement_.space().GetOrdering() == mfem::Ordering::byNODES) {
    for (int i = 0; i < scalar_size; ++i) {
      x_[i] = displacement_gf[i] + reference_nodes_[i];
      y_[i] = displacement_gf[scalar_size + i] + reference_nodes_[scalar_size + i];
      if (mesh_.Dimension() == 3) {
        z_[i] = displacement_gf[scalar_size * 2 + i] + reference_nodes_[scalar_size * 2 + i];
      } else {
        z_[i] = 0.0;
      }
    }
    // If the ordering is by vdim, it is stacked x1, y1, z1, x2, y2, z2...
  } else {
    for (int i = 0; i < scalar_size; ++i) {
      x_[i] = displacement_gf[mesh_.Dimension() * i] + reference_nodes_[mesh_.Dimension() * i];
      y_[i] = displacement_gf[mesh_.Dimension() * i + 1] + reference_nodes_[mesh_.Dimension() * i + 1];
      if (mesh_.Dimension() == 3) {
        z_[i] = displacement_gf[mesh_.Dimension() * i + 2] + reference_nodes_[mesh_.Dimension() * i + 2];
      } else {
        z_[i] = 0.0;
      }
    }
  }

  /// Update the Tribol force vectors
  p_fx_ = 0.0;
  p_fy_ = 0.0;
  p_fz_ = 0.0;

  s_fx_ = 0.0;
  s_fy_ = 0.0;
  s_fz_ = 0.0;

  // Register the two mesh sides with tribol
  tribol::registerMesh(0, num_primary_faces_, scalar_size, p_conn_.data(), bdr_type_, x_.GetData(), y_.GetData(),
                       z_.GetData());

  tribol::registerMesh(1, num_secondary_faces_, scalar_size, s_conn_.data(), bdr_type_, x_.GetData(), y_.GetData(),
                       z_.GetData());

  // Register the nodal response vectors
  tribol::registerNodalResponse(0, p_fx_.GetData(), p_fy_.GetData(), p_fz_.GetData());
  tribol::registerNodalResponse(1, s_fx_.GetData(), s_fy_.GetData(), s_fz_.GetData());

  // Register the mortar gaps and pressures
  tribol::registerRealNodalField(1, "mortar_gaps", gap_.gridFunction().GetData());
  tribol::registerRealNodalField(1, "mortar_pressures", pres_.gridFunction().GetData());

  // Set the minimum contact area fraction
  tribol::setContactAreaFrac(1.0e-6);

  // Set the tribol coupling scheme
  // note: only mortar with lagrange is currently used
  tribol::registerCouplingScheme(0, 0, 1, tribol::SURFACE_TO_SURFACE, tribol::AUTO, tribol::SINGLE_MORTAR,
                                 tribol::FRICTIONLESS, tribol::LAGRANGE_MULTIPLIER, tribol::BINNING_GRID);
#else
  SLIC_ERROR_ROOT("Tribol not enabled in this build!");

  // Quiet the complier by doing something with the mesh reference
  SLIC_INFO(axom::fmt::format("My rank = {}\n", mesh_.GetMyRank()));
#endif
}

#ifdef SERAC_USE_TRIBOL
// Update the tribol data
void ContactPhysics::update(int cycle, double time, double dt)
{
  // Update the grid function from the current newton-updated true vector
  updatePosition();

  // Call the API
  tribol::update(cycle, time, dt);

  // get the constraint matrix and finalize it
  tribol::getMfemSparseMatrix(&full_constraint_matrix_, 0);
  full_constraint_matrix_->Finalize();

  // This is the number of scalar dofs in the mesh
  int scalar_size = displacement_.space().GetNDofs();

  mfem::ParGridFunction force_gf = force_.gridFunction();

  // Update the forces and jacobian
  if (force_.space().GetOrdering() == mfem::Ordering::byNODES) {
    for (int i = 0; i < scalar_size; ++i) {
      force_gf[i]               = p_fx_[i] + s_fx_[i];
      force_gf[scalar_size + i] = p_fy_[i] + s_fy_[i];
      if (mesh_.Dimension() == 3) {
        force_gf[2 * scalar_size + i] = p_fz_[i] + s_fz_[i];
      }
    }
  } else {
    for (int i = 0; i < scalar_size; ++i) {
      force_gf[mesh_.Dimension() * i]     = p_fx_[i] + s_fx_[i];
      force_gf[mesh_.Dimension() * i + 1] = p_fy_[i] + s_fy_[i];
      if (mesh_.Dimension() == 3) {
        force_gf[mesh_.Dimension() * i + 2] = p_fz_[i] + s_fz_[i];
      }
    }
  }

  force_.setFromGridFunction(force_gf);

  mfem::ParGridFunction& gap_gf = gap_.gridFunction();
  gap_.setFromGridFunction(gap_gf);

  mfem::ParGridFunction& pressure_gf = pres_.gridFunction();
  pres_.setFromGridFunction(pressure_gf);

  // Extract the reduced secondary node jacobain contributions and put them
  // in a separate constraint sparse matrix
  constraint_matrix_ = std::make_unique<mfem::SparseMatrix>(s_nodes_.size(), displacement_.space().GetTrueVSize());
  constraint_rhs_    = 0.0;

  int num_disp_dofs = displacement_.space().GetTrueVSize();
  int mesh_dim      = displacement_.mesh().Dimension();

  int current_constraint = 0;

  // Loop over all of the constraints (i.e. secondary nodes)
  // Note: we assume all secondary nodes are active
  for (int s_node : s_nodes_) {
    // Calculate the offset for the secondary nodes in the tribol-returned monolithic matrix
    int offset = num_disp_dofs + s_node;

    // Get the number of entries in the current secondary node row
    int rowsize = full_constraint_matrix_->RowSize(offset);
    if (rowsize > 0) {
      int*    cols = full_constraint_matrix_->GetRowColumns(offset);
      double* vals = full_constraint_matrix_->GetRowEntries(offset);
      for (int j = 0; j < rowsize; ++j) {
        // Calculate the entry location in the reduced constraint matrix
        int node       = cols[j] / displacement_.mesh().Dimension();
        int active_dim = cols[j] % displacement_.mesh().Dimension();

        if (displacement_.space().GetOrdering() == mfem::Ordering::byNODES) {
          constraint_matrix_->Set(current_constraint, scalar_size * active_dim + node, vals[j]);
        } else {
          constraint_matrix_->Set(current_constraint, node * mesh_dim + active_dim, vals[j]);
        }
      }
      ++current_constraint;
    }
  }

  // Finalize the sparse matrix
  constraint_matrix_->Finalize();
}

#else
void ContactPhysics::update(int, double, double) { SLIC_ERROR_ROOT("Tribol not enabled in this build!"); }
#endif

}  // namespace serac
