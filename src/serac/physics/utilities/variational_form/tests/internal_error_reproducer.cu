// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <tuple>
auto foo(int a, int b) { return std::tuple<int,int>{a, b}; }
int main() { foo(3, 2); }
