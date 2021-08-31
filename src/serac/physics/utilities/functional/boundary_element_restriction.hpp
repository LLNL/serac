#pragma once

#include "mfem.hpp"

namespace detail {

  struct elem_info{
    int global_row;
    int global_col;
    int local_row;
    int local_col;
    int element_id;
    int sign;
    bool on_boundary;
  };

  // for sorting lexicographically by {global_row, global_col}
  inline bool operator<(const elem_info & x, const elem_info & y) {
    return (x.global_row < y.global_row) || (x.global_row == y.global_row && x.global_col < y.global_col);
  }

  inline bool operator!=(const elem_info & x, const elem_info & y) {
    return (x.global_row != y.global_row) || (x.global_col != y.global_col);
  }

  inline int get_sign(int i) { return (i >= 0) ? 1 : -1; }
  inline int get_index(int i) { return (i >= 0) ? i : - 1 - i; }

  struct signed_index{
    int index;
    int sign;
    operator int(){ return index; }
  };

  inline void apply_permutation(mfem::Array<int> & input, const mfem::Array<int> & permutation) {
    auto output = input;
    for (int i = 0; i < permutation.Size(); i++) {
      if (permutation[i] >= 0) {
        output[i] = input[permutation[i]];
      } else {
        output[i] = -input[-permutation[i]-1]-1;
      }
    }
    input = output;
  }

}

namespace serac {

struct BoundaryElementRestriction : public mfem::Operator {
  BoundaryElementRestriction(mfem::FiniteElementSpace & fespace);

  void Mult(const mfem::Vector & L, mfem::Vector & E) const override;
  void MultTranspose(const mfem::Vector & E, mfem::Vector & L) const override;

  std::vector< int > dof_ids;
  std::vector< int > elem_to_dof_id_offsets;
};

}
