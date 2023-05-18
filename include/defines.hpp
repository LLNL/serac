#pragma once

#define SERAC_HOST_DEVICE
#define SERAC_SUPPRESS_NVCC_HOSTDEVICE_WARNING

//#if __CUDAVER__ >= 75000
//#define SERAC_SUPPRESS_NVCC_HOSTDEVICE_WARNING #pragma nv_exec_check_disable
//#else
//#define SERAC_SUPPRESS_NVCC_HOSTDEVICE_WARNING #pragma hd_warning_disable
//#endif

namespace detail {
  template <typename T>
  struct always_false : std::false_type {};
}

template <int i>
struct integral_constant {
  SERAC_HOST_DEVICE constexpr operator int() { return i; }
};

SERAC_SUPPRESS_NVCC_HOSTDEVICE_WARNING
template <typename lambda, int... i>
SERAC_HOST_DEVICE constexpr void for_constexpr(lambda&& f, integral_constant<i>... args)
{
  f(args...);
}

SERAC_SUPPRESS_NVCC_HOSTDEVICE_WARNING
template <int... n, typename lambda, typename... arg_types>
SERAC_HOST_DEVICE constexpr void for_constexpr(lambda&& f, std::integer_sequence<int, n...>, arg_types... args)
{
  (for_constexpr(f, args..., integral_constant<n>{}), ...);
}

template <int... n, typename lambda>
SERAC_HOST_DEVICE constexpr void for_constexpr(lambda&& f)
{
for_constexpr(f, std::make_integer_sequence<int, n>{}...);
}