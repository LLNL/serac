#include "axom/core/utilities/Timer.hpp"

#include "serac/numerics/functional/tensor.hpp"

#define MFEM_HOST_DEVICE __host__ __device__

namespace mfem
{

/// A Class to compute the real index from the multi-indices of a tensor
template <int N, int Dim, typename T, typename... Args>
class TensorInd
{
public:
   MFEM_HOST_DEVICE
   static inline int result(const int* sizes, T first, Args... args)
   {
      return first + sizes[N - 1] * TensorInd < N + 1, Dim, Args... >
             ::result(sizes, args...);
   }
};

// Terminal case
template <int Dim, typename T, typename... Args>
class TensorInd<Dim, Dim, T, Args...>
{
public:
   MFEM_HOST_DEVICE
   static inline int result(const int* sizes, T first, Args... args)
   {
      return first;
   }
};


/// A class to initialize the size of a Tensor
template <int N, int Dim, typename T, typename... Args>
class Init
{
public:
   MFEM_HOST_DEVICE
   static inline int result(int* sizes, T first, Args... args)
   {
      sizes[N - 1] = first;
      return first * Init < N + 1, Dim, Args... >::result(sizes, args...);
   }
};

// Terminal case
template <int Dim, typename T, typename... Args>
class Init<Dim, Dim, T, Args...>
{
public:
   MFEM_HOST_DEVICE
   static inline int result(int* sizes, T first, Args... args)
   {
      sizes[Dim - 1] = first;
      return first;
   }
};


/// A basic generic Tensor class, appropriate for use on the GPU
template<int Dim, typename Scalar = double>
class DeviceTensor
{
protected:
   int capacity;
   Scalar *data;
   int sizes[Dim];

public:
   /// Default constructor
   DeviceTensor() = delete;

   /// Constructor to initialize a tensor from the Scalar array data_
   template <typename... Args> MFEM_HOST_DEVICE
   DeviceTensor(Scalar* data_, Args... args)
   {
      static_assert(sizeof...(args) == Dim, "Wrong number of arguments");
      // Initialize sizes, and compute the number of values
      const long int nb = Init<1, Dim, Args...>::result(sizes, args...);
      capacity = nb;
      data = (capacity > 0) ? data_ : NULL;
   }

   /// Copy constructor
   MFEM_HOST_DEVICE DeviceTensor(const DeviceTensor<Dim, Scalar> & t)
   {
      capacity = t.capacity;
      for (int i = 0; i < Dim; ++i)
      {
         sizes[i] = t.sizes[i];
      }
      data = t.data;
   }

   /// Conversion to `Scalar *`.
   MFEM_HOST_DEVICE inline operator Scalar *() const { return data; }

   /// Const accessor for the data
   template <typename... Args> MFEM_HOST_DEVICE inline
   Scalar& operator()(Args... args) const
   {
      static_assert(sizeof...(args) == Dim, "Wrong number of arguments");
      return data[ TensorInd<1, Dim, Args...>::result(sizes, args...) ];
   }

   /// Subscript operator where the tensor is viewed as a 1D array.
   MFEM_HOST_DEVICE inline Scalar& operator[](int i) const
   {
      return data[i];
   }
};


/** @brief Wrap a pointer as a DeviceTensor with automatically deduced template
    parameters */
template <typename T, typename... Dims>
inline DeviceTensor<sizeof...(Dims),T> Reshape(T *ptr, Dims... dims)
{
   return DeviceTensor<sizeof...(Dims),T>(ptr, dims...);
}

} // mfem namespace

__global__ void bandwidth_test_raw_ptr(double * output, double * input, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) { output[i] = input[i]; }
}

template <int dim, int q>
__global__ void bandwidth_test_DeviceTensor(mfem::DeviceTensor<5, double>       output,
                                            const mfem::DeviceTensor<5, double> input, int num_elements)
{
    int element = blockIdx.x;

    const double O11 = input(threadIdx.x,threadIdx.y,threadIdx.z,0,element);
    const double O12 = input(threadIdx.x,threadIdx.y,threadIdx.z,1,element);
    const double O13 = input(threadIdx.x,threadIdx.y,threadIdx.z,2,element);
    const double O21 = input(threadIdx.x,threadIdx.y,threadIdx.z,3,element);
    const double O22 = input(threadIdx.x,threadIdx.y,threadIdx.z,4,element);
    const double O23 = input(threadIdx.x,threadIdx.y,threadIdx.z,5,element);
    const double O31 = input(threadIdx.x,threadIdx.y,threadIdx.z,6,element);
    const double O32 = input(threadIdx.x,threadIdx.y,threadIdx.z,7,element);
    const double O33 = input(threadIdx.x,threadIdx.y,threadIdx.z,8,element);

    output(threadIdx.x,threadIdx.y,threadIdx.z,0,element) = O11;
    output(threadIdx.x,threadIdx.y,threadIdx.z,1,element) = O12;
    output(threadIdx.x,threadIdx.y,threadIdx.z,2,element) = O13;
    output(threadIdx.x,threadIdx.y,threadIdx.z,3,element) = O21;
    output(threadIdx.x,threadIdx.y,threadIdx.z,4,element) = O22;
    output(threadIdx.x,threadIdx.y,threadIdx.z,5,element) = O23;
    output(threadIdx.x,threadIdx.y,threadIdx.z,6,element) = O31;
    output(threadIdx.x,threadIdx.y,threadIdx.z,7,element) = O32;
    output(threadIdx.x,threadIdx.y,threadIdx.z,8,element) = O33;

}

template <int dim, int q>
__global__ void bandwidth_test_DeviceTensor(mfem::DeviceTensor<6, double>       output,
                                            const mfem::DeviceTensor<6, double> input, int num_elements)
{
    int element = blockIdx.x;

    const double O11 = input(threadIdx.x,threadIdx.y,threadIdx.z,0,0,element);
    const double O12 = input(threadIdx.x,threadIdx.y,threadIdx.z,1,0,element);
    const double O13 = input(threadIdx.x,threadIdx.y,threadIdx.z,2,0,element);
    const double O21 = input(threadIdx.x,threadIdx.y,threadIdx.z,0,1,element);
    const double O22 = input(threadIdx.x,threadIdx.y,threadIdx.z,1,1,element);
    const double O23 = input(threadIdx.x,threadIdx.y,threadIdx.z,2,1,element);
    const double O31 = input(threadIdx.x,threadIdx.y,threadIdx.z,0,2,element);
    const double O32 = input(threadIdx.x,threadIdx.y,threadIdx.z,1,2,element);
    const double O33 = input(threadIdx.x,threadIdx.y,threadIdx.z,2,2,element);

    output(threadIdx.x,threadIdx.y,threadIdx.z,0,0,element) = O11;
    output(threadIdx.x,threadIdx.y,threadIdx.z,1,0,element) = O12;
    output(threadIdx.x,threadIdx.y,threadIdx.z,2,0,element) = O13;
    output(threadIdx.x,threadIdx.y,threadIdx.z,0,1,element) = O21;
    output(threadIdx.x,threadIdx.y,threadIdx.z,1,1,element) = O22;
    output(threadIdx.x,threadIdx.y,threadIdx.z,2,1,element) = O23;
    output(threadIdx.x,threadIdx.y,threadIdx.z,0,2,element) = O31;
    output(threadIdx.x,threadIdx.y,threadIdx.z,1,2,element) = O32;
    output(threadIdx.x,threadIdx.y,threadIdx.z,2,2,element) = O33;
}

template <int dim, int q>
__global__ void bandwidth_test_make_tensor(mfem::DeviceTensor<6, double>       output,
                                           const mfem::DeviceTensor<6, double> input, int num_elements)
{
    int element = blockIdx.x;

    serac::tensor<double,dim,dim> J;
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            J[i][j] = input(threadIdx.x,threadIdx.y,threadIdx.z,j,i,element);
        }
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            output(threadIdx.x,threadIdx.y,threadIdx.z,j,i,element) = J[i][j];
        }
    }
}

int main() {

    constexpr int q = 4;
    constexpr int dim = 3;
    constexpr int num_elements = 65536;

    std::cout << q * q * q * dim * dim * size_t(num_elements) * 16 << std::endl; 

    double * inputs;
    double * outputs;
    cudaMalloc(&inputs, sizeof(double) * num_elements * dim * dim * q * q * q);
    cudaMalloc(&outputs, sizeof(double) * num_elements * dim * dim * q * q * q);

    mfem::DeviceTensor<5, double> inputs5D(inputs, q, q, q, dim * dim, num_elements);
    mfem::DeviceTensor<6, double> inputs6D(inputs, q, q, q, dim, dim, num_elements);

    mfem::DeviceTensor<5, double> outputs5D(outputs, q, q, q, dim * dim, num_elements);
    mfem::DeviceTensor<6, double> outputs6D(outputs, q, q, q, dim, dim, num_elements);

    for (int i = 0; i < 16; i++) {
        {
            int n = num_elements * dim * dim * q * q * q;
            int blocksize = 128;
            int gridsize = (n + blocksize - 1) / blocksize;
            bandwidth_test_raw_ptr<<<gridsize, blocksize>>>(inputs, outputs, n);
        }

        dim3 blocksize = {q, q, q};
        dim3 gridsize = num_elements;
        bandwidth_test_DeviceTensor<dim, q><<<gridsize, blocksize>>>(inputs5D, outputs5D, num_elements);
        bandwidth_test_DeviceTensor<dim, q><<<gridsize, blocksize>>>(inputs6D, outputs6D, num_elements);
        bandwidth_test_make_tensor<dim, q><<<gridsize, blocksize>>>(inputs6D, outputs6D, num_elements);

        cudaMemcpy(inputs, outputs, sizeof(double) * num_elements * dim * dim * q * q * q, cudaMemcpyDeviceToDevice);
    }

    cudaFree(inputs);
    cudaFree(outputs);

}
