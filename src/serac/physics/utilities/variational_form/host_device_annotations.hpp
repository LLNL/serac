// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#if defined(__CUDACC__)
#define SERAC_HOST_DEVICE __host__ __device__
#else
#define SERAC_HOST_DEVICE
#endif
