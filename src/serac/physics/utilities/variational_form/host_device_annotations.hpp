#pragma once

#if defined(__CUDACC__)
//#define SERAC_HOST_DEVICE __host__ __device__
#define SERAC_HOST_DEVICE
#else
#define SERAC_HOST_DEVICE
#endif