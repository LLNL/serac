#pragma once

#include "mfem.hpp"
#include "axom/core.hpp"

#include <string>
#include <fstream>
#include <iomanip>
#include <vector>

#include "serac/numerics/functional/dof_numbering.hpp"

template <typename T>
void write_to_file(std::vector<T> v, std::string filename)
{
  std::ofstream outfile(filename);
  for (int i = 0; i < v.size(); i++) {
    outfile << v[i] << std::endl;
  }
  outfile.close();
}

void write_to_file(mfem::Vector v, std::string filename)
{
  std::ofstream outfile(filename);
  for (int i = 0; i < v.Size(); i++) {
    outfile << v[i] << std::endl;
  }
  outfile.close();
}

void write_to_file(mfem::SparseMatrix A, std::string filename)
{
  std::ofstream outfile(filename);
  A.PrintMM(outfile);
  outfile.close();
}

std::ostream& operator<<(std::ostream& out, DoF dof)
{
  out << "{" << dof.index() << ", " << dof.sign() << ", " << dof.orientation() << "}";
  return out;
}

std::ostream& operator<<(std::ostream& out, serac::SignedIndex i)
{
  out << "{" << i.index_ << ", " << i.sign_ << "}";
  return out;
}

template <typename T>
void write_to_file(axom::Array<T, 2, axom::MemorySpace::Host> arr, std::string filename)
{
  std::ofstream outfile(filename);

  for (axom::IndexType i = 0; i < arr.shape()[0]; i++) {
    outfile << "{";
    for (axom::IndexType j = 0; j < arr.shape()[1]; j++) {
      outfile << arr(i, j);
      if (j < arr.shape()[1] - 1) outfile << ", ";
    }
    outfile << "}\n";
  }

  outfile.close();
}

template <typename T>
void write_to_file(axom::Array<T, 3, axom::MemorySpace::Host> arr, std::string filename)
{
  std::ofstream outfile(filename);

  outfile << std::setprecision(16);

  for (axom::IndexType i = 0; i < arr.shape()[0]; i++) {
    outfile << "{";
    for (axom::IndexType j = 0; j < arr.shape()[1]; j++) {
      outfile << "{";
      for (axom::IndexType k = 0; k < arr.shape()[2]; k++) {
        outfile << arr(i, j, k);
        if (k < arr.shape()[2] - 1) outfile << ", ";
      }
      outfile << "}";
      if (j < arr.shape()[1] - 1) outfile << ", ";
    }
    outfile << "}\n";
  }

  outfile.close();
}
