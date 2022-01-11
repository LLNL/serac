// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <cstdio>

template <int n>
void print()
{
  printf("%d ", n);
}

template <int... n, typename lambda_type>
auto print_each_and_invoke(lambda_type f)
{
  (print<n>(), ...);
  return f();
}

int main()
{
  return print_each_and_invoke<3, 4, 5>([]() { return 42; });
}
