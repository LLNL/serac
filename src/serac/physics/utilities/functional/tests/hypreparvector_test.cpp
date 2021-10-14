// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <unistd.h>
#include <fstream>
#include <iostream>
#include <thread>
#include <chrono>

#include "mfem.hpp"

#include <gtest/gtest.h>

using namespace std::chrono_literals;

int main(int argc, char* argv[])
{
  int num_procs, myid;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  int entries_per_proc = 6;
  int total_entries = entries_per_proc * num_procs;

  mfem::Vector x_global(total_entries);
  mfem::Vector y_global(total_entries);

  for (int i = 0; i < total_entries; i++) {
    x_global(i) = sin(i);
    y_global(i) = sin(i * i + 0.3);
  }

  double x_local_data[entries_per_proc];
  double y_local_data[entries_per_proc];
  for (int i = 0; i < entries_per_proc; i++) {
    x_local_data[i] = x_global(i + myid * entries_per_proc);
    y_local_data[i] = y_global(i + myid * entries_per_proc);
  }

  HYPRE_BigInt col[2] = {myid * entries_per_proc, (myid + 1) * entries_per_proc};
  mfem::HypreParVector x_local(MPI_COMM_WORLD, total_entries, x_local_data, col);
  mfem::HypreParVector y_local(MPI_COMM_WORLD, total_entries, y_local_data, col);

  double xTx = InnerProduct(x_local, x_local);

  if (myid == 0) {
    printf("before: \n");
    printf("x_global * x_global: %f\n"
           "x_local * x_local: %f\n\n",
           InnerProduct(x_global, x_global), xTx);
  }

  auto orthogonalize = [](mfem::Vector & x, mfem::Vector & y) {
    x -= ((x * y) / (y * y)) * y;
    x -= (dot(x,y) / dot(y,y)) * y;
    x.Add(-InnerProduct(x, y) / InnerProduct(y, y), y);
  };

  orthogonalize(x_global, y_global);
  orthogonalize(x_local, y_local);

  xTx = InnerProduct(x_local, x_local);

  if (myid == 0) {
    printf("after: \n");
    printf("x_global * x_global: %f\n"
           "x_local * x_local: %f\n\n",
           InnerProduct(x_global, x_global), xTx);
  }

  {
    mfem::Vector U = x_global;

    U(0) = 42.0;
    if (myid == 0) std::cout << U(0) << std::endl;

    auto V = U;
    if (myid == 0) std::cout << V(0) << std::endl;
  }

  if (myid == 0) std::cout << std::endl;

  {
    mfem::HypreParVector U = x_local;

    U(0) = 42.0;
    if (myid == 0) std::cout << U(0) << std::endl;

    auto V = U;
    if (myid == 0) std::cout << V(0) << std::endl;
  }

  MPI_Finalize();
}
