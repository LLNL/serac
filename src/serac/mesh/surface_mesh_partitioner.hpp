// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "serac/mesh/mesh_utils_base.hpp"

namespace mfem
{

// This is highly copied from the MFEM SurfaceRepartition branch 
class SurfaceRepartition : public Mesh
{
public:

   SurfaceRepartition(SurfaceRepartition &&SurfaceRepartition) = default;

   SurfaceRepartition& operator=(SurfaceRepartition &&SurfaceRepartition) = delete;

   SurfaceRepartition& operator=(const SurfaceRepartition &SurfaceRepartition) = delete;

   SurfaceRepartition() = delete;

   // Create a domain SurfaceRepartition from it's parent.
   //
   // The attributes have to mark exactly one connected subset of the parent
   // Mesh.
   static SurfaceRepartition CreateFromSurface(ParMesh &parent);

   const Mesh* GetParent() { return &parent_; }

   const Array<int>& GetParentElementIDMap() const
   {
      return parent_element_ids_;
   }
   // void GetParentElementIDMap(Array<int>& map);

   // const Array<int>& GetParentEdgeIDMap() const;
   // void GetParentEdgeIDMap(Array<int>& map);

   const Array<int>& GetParentVertexIDMap() const
   {
      return parent_vertex_ids_;
   }
   // void GetParentVertexIDMap(Array<int>& map);

   // Transfer the dofs of a GridFunction on a SurfaceRepartition to a GridFunction on the
   // Mesh.
   static void TransferToSerial(ParGridFunction &src, GridFunction &dst);

   static void TransferToSerial(QuadratureFunction &src, QuadratureFunction &dst);

   static void TransferToParallel(GridFunction &src, ParGridFunction &dst);

   static void TransferToParallel(QuadratureFunction &src, QuadratureFunction &dst);

   ~SurfaceRepartition();

private:

   // The parent Mesh
   ParMesh &parent_;

   Array<int> attributes_;
   Array<int> element_ids_;

   // Mapping from SurfaceRepartition element ids (index of the array), to
   // the parent element ids.
   Array<int> parent_element_ids_;

   // // Mapping from SurfaceRepartition edge ids (index of the array), to
   // // the parent edge ids.
   // Array<int> parent_edge_ids_;

   // Mapping from SurfaceRepartition vertex ids (index of the array), to
   // the parent vertex ids.
   Array<int> parent_vertex_ids_;

};
};
