#include "serac/physics/utilities/functional/boundary_element_restriction.hpp"

auto decode_sign_and_index(int i) {
  if (i >= 0) { return ::detail::signed_index{1, i}; } 
  else { return ::detail::signed_index{-1, -1 - i}; }
}

namespace serac {

BoundaryElementRestriction::BoundaryElementRestriction(mfem::FiniteElementSpace & fes) {

  int vdim = fes.GetVDim();
  mfem::Array< int > belem_dof_ids;
  elem_to_dof_id_offsets.push_back(0);

  std::ofstream outfile("bdr_element_restriction.txt");

  for (int b = 0; b < fes.GetNBE(); b++) {
    fes.GetBdrElementDofs(b, belem_dof_ids);

    if (true) {
      const mfem::Array<int> & native_to_lexicographic = dynamic_cast<const mfem::TensorBasisElement *>(fes.GetBE(b))->GetDofMap();
      ::detail::apply_permutation(belem_dof_ids, native_to_lexicographic);
    }

    for (int i = 0; i < belem_dof_ids.Size(); i++) {
      for (int j = 0; j < vdim; j++) {
        int vdof = fes.DofToVDof(::detail::get_index(belem_dof_ids[i]), j);

        dof_ids.push_back(vdof);
        outfile << vdof << " ";
      }
      outfile << std::endl;
    }

    elem_to_dof_id_offsets.push_back(int(dof_ids.size() * vdim));
  }

  outfile.close();

  // size of "E" vector
  height = int(dof_ids.size());

  // size of "L" vector
  width = fes.GetVSize();

}

void BoundaryElementRestriction::Mult(const mfem::Vector & L, mfem::Vector & E) const {
  for (int i = 0; i < height; i++) { 
    auto [sign, index] = decode_sign_and_index(dof_ids[i]);
    E[i] = sign * L[index]; 
  }
}

void BoundaryElementRestriction::MultTranspose(const mfem::Vector & E, mfem::Vector & L) const {
  L = 0.0;
  for (int i = 0; i < width; i++) { 
    auto [sign, index] = decode_sign_and_index(dof_ids[i]);
    L[index] += E[i]; 
  }
}

}
