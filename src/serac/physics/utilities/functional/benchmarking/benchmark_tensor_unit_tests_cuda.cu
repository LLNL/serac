// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/utilities/functional/tensor.hpp"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

template <int n>
void custom_assert(bool condition, const char (&message)[n])
{
  if (condition == false) {
    printf("error: %s", message);
  }
}

template <typename TensorType>
__global__ void set_tensor(int N, TensorType tensor, TensorType* data)
{
  int id          = blockIdx.x * blockDim.x + threadIdx.x;
  int grid_stride = blockDim.x * gridDim.x;
  for (int i = id; i < N; i += grid_stride) {
    data[i] = tensor;
  }
}

template <typename ... TupleATypes>
__global__ void set_SoA(int N, TupleATypes ... args)
{
  int id          = blockIdx.x * blockDim.x + threadIdx.x;
  int grid_stride = blockDim.x * gridDim.x;
  for (int i = id; i < N; i += grid_stride) {
     ([&] (auto & input)
     {
        auto member_var = std::get<0>(input);
        auto device_array = std::get<1>(input);
        device_array[i] = member_var;
     } (args), ...);
  }
}

template <typename ... TupleATypes>
void run_SoA_copy_benchmark(int N, dim3 grid, dim3 threadblock, TupleATypes ... values)
{
  std::tuple<thrust::device_vector<TupleATypes> ...> device_vectors;

  std::apply([&](auto&&... device_vector) {
     (device_vector.resize(N), ...);
  }, device_vectors);

  auto vals = std::make_tuple(values...);

    // allocate tuples
    auto T = 
        std::apply([&](auto & ... val){
            return std::apply([&](auto & ... vector) {
                ((std::cout << val << " " << vector.data() << std::endl),...);
        return std::make_tuple(std::make_tuple(val, thrust::raw_pointer_cast(vector.data()))...);
                }, device_vectors);
        }, vals);

  // run set_SoA benchmark
  set_SoA<<<grid, threadblock>>>(N, std::forward(T));

  // // set tensors and copy to output
  // benchmark_tensor_throughput<<<grid, threadblock>>>(N, d_input, d_output);

  // wait for it to finish
  cudaDeviceSynchronize();
}

template <typename TensorType>
__global__ void benchmark_tensor_throughput(int N, TensorType* input, TensorType* output)
{
  int id          = blockIdx.x * blockDim.x + threadIdx.x;
  int grid_stride = blockDim.x * gridDim.x;
  for (int i = id; i < N; i += grid_stride) {
    output[i] = input[i];
  }
}

template <typename TensorType>
void run_tensor_copy_benchmark(int N, TensorType tensor, dim3 grid, dim3 threadblock)
{
  thrust::device_vector<TensorType> input(N);
  thrust::device_vector<TensorType> output(N);

  auto d_input  = thrust::raw_pointer_cast(input.data());
  auto d_output = thrust::raw_pointer_cast(output.data());

  // initialize input
  set_tensor<<<grid, threadblock>>>(N, tensor, d_input);

  // set tensors and copy to output
  benchmark_tensor_throughput<<<grid, threadblock>>>(N, d_input, d_output);

  // wait for it to finish
  cudaDeviceSynchronize();
}

namespace benchmark {
struct double2 {
  double x_1;
  double x_2;
};

struct __align__(16) double2_16
{
  double x_1;
  double x_2;
};

/* duplicate of tensor */

template <typename T, int... n>
struct tensor;

template <typename T>
struct tensor<T> {
  using type                                  = T;
  static constexpr int              ndim      = 0;
  static constexpr int              first_dim = 1;
  SERAC_HOST_DEVICE constexpr auto& operator[](int) { return value; }
  SERAC_HOST_DEVICE constexpr auto  operator[](int) const { return value; }

  template <typename... S>
  SERAC_HOST_DEVICE constexpr auto& operator()(S...)
  {
    return value;
  }

  template <typename... S>
  SERAC_HOST_DEVICE constexpr auto operator()(S...) const
  {
    return value;
  }

  SERAC_HOST_DEVICE tensor() : value{} {}
  SERAC_HOST_DEVICE tensor(T v) : value(v) {}
  SERAC_HOST_DEVICE operator T() { return value; }
  T                 value;
};

template <typename T, int n>
struct __align__(16) tensor<T, n>
{
  using type                     = T;
  static constexpr int ndim      = 1;
  static constexpr int first_dim = n;

  template <typename S>
  SERAC_HOST_DEVICE constexpr auto& operator()(S i)
  {
    return serac::detail::get(value, i);
  }

  template <typename S>
  SERAC_HOST_DEVICE constexpr auto operator()(S i) const
  {
    return serac::detail::get(value, i);
  }

  SERAC_HOST_DEVICE constexpr auto& operator[](int i) { return value[i]; };
  SERAC_HOST_DEVICE constexpr auto  operator[](int i) const { return value[i]; };
  T                                 value[n];
};

/**
 * @brief Arbitrary-rank tensor class
 * @tparam T The scalar type of the tensor
 * @tparam first The leading dimension of the tensor
 * @tparam last The parameter pack of the remaining dimensions
 */
template <typename T, int first, int... rest>
struct tensor<T, first, rest...> {
  /**
   * @brief The scalar type
   */
  using type = T;
  /**
   * @brief The rank of the tensor
   */
  static constexpr int ndim = 1 + sizeof...(rest);
  /**
   * @brief The array of dimensions containing the shape (not the data itself)
   * Similar to numpy.ndarray.shape
   */
  static constexpr int first_dim = first;

  /**
   * @brief Retrieves the sub-tensor corresponding to the indices provided in the pack @a i
   * @param[in] i The pack of indices
   */
  template <typename... S>
  SERAC_HOST_DEVICE constexpr auto& operator()(S... i)
  {
    // FIXME: Compile-time check for <= 4 indices??
    return serac::detail::get(value, i...);
  };
  /// @overload
  template <typename... S>
  SERAC_HOST_DEVICE constexpr auto operator()(S... i) const
  {
    return serac::detail::get(value, i...);
  };

  /**
   * @brief Retrieves the "row" of the tensor corresponding to index @a i
   * @param[in] i The index to retrieve a rank - 1 tensor from
   */
  SERAC_HOST_DEVICE constexpr auto& operator[](int i) { return value[i]; };
  /// @overload
  SERAC_HOST_DEVICE constexpr auto operator[](int i) const { return value[i]; };

  /**
   * @brief The actual tensor data
   */
  tensor<T, rest...> value[first];
};


}  // namespace benchmark

struct S1 {
  int a;
  double b;
};

int main()
{
  int          N         = 10000000;  // ~ 10M
  unsigned int blocksize = 256;
  dim3         threadblock{blocksize, 1, 1};
  dim3         grid{(N + blocksize - 1) / blocksize, 1, 1};
  run_tensor_copy_benchmark(N, serac::tensor<double, >({1}), grid, threadblock);
  run_tensor_copy_benchmark(N, serac::tensor<double, 2>({1, 2}), grid, threadblock);
  run_tensor_copy_benchmark(N, serac::tensor<double, 3>({1, 2, 3}), grid, threadblock);
  run_tensor_copy_benchmark(N, serac::tensor<double, 4>({1, 2, 3, 4}), grid, threadblock);
  run_tensor_copy_benchmark(N, serac::tensor<double, 2, 2>({{{1, 2}, {2, 1}}}), grid, threadblock);
  run_tensor_copy_benchmark(N, serac::tensor<double, 3, 3>({{{1, 2, 3}, {1, 2, 3}, {1, 2, 3}}}), grid, threadblock);

  // simple memory experiments on custom types
  run_tensor_copy_benchmark(N, benchmark::double2({1, 2}), grid, threadblock);
  run_tensor_copy_benchmark(N, double2({1, 2}), grid, threadblock);
  run_tensor_copy_benchmark(N, benchmark::double2_16({1, 2}), grid, threadblock);
  run_tensor_copy_benchmark(N, benchmark::tensor<double, 2>({1, 2}), grid, threadblock);
  run_tensor_copy_benchmark(N, benchmark::tensor<double, 3>({1, 2, 3}), grid, threadblock);
  run_tensor_copy_benchmark(N, benchmark::tensor<double, 4>({1, 2, 3, 4}), grid, threadblock);
  run_tensor_copy_benchmark(N, benchmark::tensor<double, 2, 2>({{{1, 2}, {2, 1}}}), grid, threadblock);
  run_tensor_copy_benchmark(N, benchmark::tensor<double, 3, 3>({{{1, 2, 3}, {1, 2, 3}, {1, 2, 3}}}), grid, threadblock);

  // SoA benchmark
  S1 s1 { .a = 1, .b=1.234};
  run_SoA_copy_benchmark(N, grid, threadblock, s1.a, s1.b);
}

// ncu -f -o unit_test --details-all ./tests/benchmark_tensor_unit_tests_cuda
// ncu -f -o unit_test --details-all --set full ./tests/benchmark_tensor_unit_tests_cuda
// ncu -f -o unit_test --set full -k "set_tensor|benchmark" ./tests/benchmark_tensor_unit_tests_cuda
