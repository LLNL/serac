#pragma once

#if defined(__CUDACC__)
#define SERAC_HOST_DEVICE __host__ __device__
#else
#define SERAC_HOST_DEVICE
#endif
