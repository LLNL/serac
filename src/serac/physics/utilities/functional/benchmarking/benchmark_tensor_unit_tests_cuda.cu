// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/utilities/functional/tensor.hpp"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

/**
 * @brief Kernel to set an array of typed data on a GPU with the provided tensor
 * @param N the number of array elements
 * @param tensor the value that each element should be initialized to
 * @param data The gpu typed array
 */
template <typename StructType>
__global__ void set_struct(int N, StructType tensor, StructType* data)
{
  int id          = blockIdx.x * blockDim.x + threadIdx.x;
  int grid_stride = blockDim.x * gridDim.x;
  for (int i = id; i < N; i += grid_stride) {
    data[i] = tensor;
  }
}

/**
 * @brief Kernel to set a structure of GPU arrays of typed data
 *
 * The args argument takes in a tuple of tuples. Each of the inner tuples contains firstly
 * the value to set, and secondly the GPU array pointer within the Structure of Arrays that
 * we would like to initialize using the first tuple value.
 *
 * @param N the number of elements in each array within the structure
 * @param args A std::tuple of pairs of (value to set, gpu pointer array start)
 */
template <typename... TupleATypes>
__global__ void set_SoA(int N, std::tuple<TupleATypes...> args)
{
  int id          = blockIdx.x * blockDim.x + threadIdx.x;
  int grid_stride = blockDim.x * gridDim.x;
  for (int i = id; i < N; i += grid_stride) {
    std::apply([&](auto&... pair) { ((std::get<1>(pair)[i] = std::get<0>(pair)), ...); }, args);
  }
}

/**
 * @brief A kernel bandwidth test copying tuple values between two Structures of Arrays on GPUs.
 *
 * The copyTuples argument takes in a tuple of tuples. Each of the inner tuples contains
 * firstly the output array within the structure of arrays, and the second argument is the
 * array in the Structure of Arrays to copy from. The elements are then copied element by
 * element.
 *
 * @param N the number of elements in each of the arrays in the Structure of Arrays
 * @param copyTuples
 */
template <typename... TupleATypes>
__global__ void benchmark_SoA_throughput(int N, std::tuple<TupleATypes...> copyTuples)
{
  int id          = blockIdx.x * blockDim.x + threadIdx.x;
  int grid_stride = blockDim.x * gridDim.x;
  for (int i = id; i < N; i += grid_stride) {
    std::apply([&](auto&... pair) { ((std::get<0>(pair)[i] = std::get<1>(pair)[i]), ...); }, copyTuples);
  }
}

/**
 * @brief The kernel bandwidth test to copy from a typed input array on the GPU to a typed output arary on the GPU
 *
 * @param N the number of elements within the input and output array
 * @param input the GPU input array pointer
 * @param output the GPU output array pointer
 */
template <typename StructType>
__global__ void benchmark_struct_throughput(int N, StructType* input, StructType* output)
{
  int id          = blockIdx.x * blockDim.x + threadIdx.x;
  int grid_stride = blockDim.x * gridDim.x;
  for (int i = id; i < N; i += grid_stride) {
    output[i] = input[i];
  }
}

/**
 * @brief A method to drive the structures of arrays benchmark
 *
 * Thrust is used to allocate GPU arrays
 *
 * @param N the number of elements
 * @param grid The GPU grid launch configuration
 * @param threadblock The threadblock configuration to use
 * @param values A list of values corresponding to different elements within the structure of arrays to initialize each
 * of the arrays.
 */
template <typename... TupleATypes>
void run_SoA_copy_benchmark(int N, dim3 grid, dim3 threadblock, TupleATypes... values)
{
  std::tuple<thrust::device_vector<TupleATypes>...> device_vectors;
  std::tuple<thrust::device_vector<TupleATypes>...> output_device_vectors;

  // allocate input and output structure of arrays  on the GPU
  std::apply([&](auto&&... device_vector) { (device_vector.resize(N), ...); }, device_vectors);
  std::apply([&](auto&&... output_device_vector) { (output_device_vector.resize(N), ...); }, output_device_vectors);

  auto vals = std::make_tuple(values...);

  // allocate input tuples
  auto setT = std::apply(
      [&](auto&... val) {
        return std::apply(
            [&](auto&... vector) {
              return std::make_tuple(std::make_tuple(val, thrust::raw_pointer_cast(vector.data()))...);
            },
            device_vectors);
      },
      vals);

  // allocate output tuples
  auto copyT = std::apply(
      [&](auto&... out_vector) {
        return std::apply(
            [&](auto&... vector) {
              return std::make_tuple(std::make_tuple(thrust::raw_pointer_cast(out_vector.data()),
                                                     thrust::raw_pointer_cast(vector.data()))...);
            },
            device_vectors);
      },
      output_device_vectors);

  // run set_SoA benchmark
  set_SoA<<<grid, threadblock>>>(N, setT);

  // set tensors and copy to output
  benchmark_SoA_throughput<<<grid, threadblock>>>(N, copyT);

  // wait for it to finish
  cudaDeviceSynchronize();
}

/**
 * @brief Drives the Arrays of structures bandwidth copy test
 *
 * @param N the number of elements
 * @param tensor The struct value to initialize the Array of structures
 * @param grid The grid configuration to execute with
 * @param threadblock The thread configuration to execut with.
 */
template <typename StructType>
void run_tensor_copy_benchmark(int N, StructType tensor, dim3 grid, dim3 threadblock)
{
  thrust::device_vector<StructType> input(N);
  thrust::device_vector<StructType> output(N);

  auto d_input  = thrust::raw_pointer_cast(input.data());
  auto d_output = thrust::raw_pointer_cast(output.data());

  // initialize input
  set_struct<<<grid, threadblock>>>(N, tensor, d_input);

  // set tensors and copy to output
  benchmark_struct_throughput<<<grid, threadblock>>>(N, d_input, d_output);

  // wait for it to finish
  cudaDeviceSynchronize();
}

namespace benchmark {
struct double2 {
  double x_1;
  double x_2;
};

/// This is a special CUDA aligned type, which may allow for the compiler to issue instructions that can access more
/// memory efficiently.
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

// Let's try aligning each row
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

// A custom struct type for Structure of Arrays experiments
struct S1 {
  int    a;
  double b;
};

int main()
{
  int          N         = 10000000;  // ~ 10M elements
  unsigned int blocksize = 256;
  dim3         threadblock{blocksize, 1, 1};
  dim3         grid{(N + blocksize - 1) / blocksize, 1, 1};

  // Use standard serac::tensor types
  run_tensor_copy_benchmark(N, serac::tensor<double, >({1}), grid, threadblock);
  run_tensor_copy_benchmark(N, serac::tensor<double, 2>({1, 2}), grid, threadblock);
  run_tensor_copy_benchmark(N, serac::tensor<double, 3>({1, 2, 3}), grid, threadblock);
  run_tensor_copy_benchmark(N, serac::tensor<double, 4>({1, 2, 3, 4}), grid, threadblock);
  run_tensor_copy_benchmark(N, serac::tensor<double, 2, 2>({{{1, 2}, {2, 1}}}), grid, threadblock);
  run_tensor_copy_benchmark(N, serac::tensor<double, 3, 3>({{{1, 2, 3}, {1, 2, 3}, {1, 2, 3}}}), grid, threadblock);

  // simple memory experiments on custom types with alignment
  run_tensor_copy_benchmark(N, benchmark::double2({1, 2}), grid, threadblock);
  run_tensor_copy_benchmark(N, double2({1, 2}), grid, threadblock);
  run_tensor_copy_benchmark(N, benchmark::double2_16({1, 2}), grid, threadblock);
  run_tensor_copy_benchmark(N, benchmark::tensor<double, 2>({1, 2}), grid, threadblock);
  run_tensor_copy_benchmark(N, benchmark::tensor<double, 3>({1, 2, 3}), grid, threadblock);
  run_tensor_copy_benchmark(N, benchmark::tensor<double, 4>({1, 2, 3, 4}), grid, threadblock);
  run_tensor_copy_benchmark(N, benchmark::tensor<double, 2, 2>({{{1, 2}, {2, 1}}}), grid, threadblock);
  run_tensor_copy_benchmark(N, benchmark::tensor<double, 3, 3>({{{1, 2, 3}, {1, 2, 3}, {1, 2, 3}}}), grid, threadblock);

  // Structure of Arrays (SoA) benchmark
  S1 s1{.a = 1, .b = 1.234};
  run_SoA_copy_benchmark(N, grid, threadblock, s1.a, s1.b);

  serac::tensor<double, 2> tensor2({1, 2});
  run_SoA_copy_benchmark(N, grid, threadblock, tensor2(0), tensor2(1));

  benchmark::tensor<double, 3, 3> tensor3({{{1, 2, 3}, {1, 2, 3}, {1, 2, 3}}});
  run_SoA_copy_benchmark(N, grid, threadblock, tensor3(1, 1), tensor3(1, 2), tensor3(1, 3), tensor3(2, 1),
                         tensor3(2, 2), tensor3(2, 3), tensor3(3, 1), tensor3(3, 2), tensor3(3, 3));
}

// Sample commands used to profile this benchmark
// ncu -f -o unit_test --details-all ./tests/benchmark_tensor_unit_tests_cuda
// ncu -f -o unit_test --details-all --set full ./tests/benchmark_tensor_unit_tests_cuda
// ncu -f -o unit_test --set full -k "set_tensor|benchmark" ./tests/benchmark_tensor_unit_tests_cuda
