#include <iostream>                                                              
                                                                                 
template <typename T>                                                            
struct QuadratureData {                                                          
  QuadratureData(int nelem, int nquad) : stride(nquad) {                         
    cudaMallocManaged(&ptr, sizeof(T) * nelem * nquad);                          
  };                                                                             
                                                                                 
  void destroy() { cudaFree(ptr); }                                              
                                                                                 
  __host__ __device__ T & operator()(int e, int q) { return ptr[e * stride + q]; }
  T * ptr;                                                                       
  int stride;                                                                    
};                                                                               
                                                                                 
template< typename T >                                                           
__global__ void fill(QuadratureData<T> output, int num_elements, int num_quadrature_points) {
  int elem_id = threadIdx.x + blockIdx.x * blockDim.x;                           
  int quad_id = threadIdx.y;                                                     
  if (elem_id < num_elements && quad_id < num_quadrature_points) {               
    output(elem_id, quad_id) = elem_id * elem_id + quad_id;                      
  }                                                                              
}                                                                                
                                                                                 
template< typename T >                                                           
__global__ void copy(QuadratureData<T> destination, QuadratureData<T> source, int num_elements, int num_quadrature_points) {
  int elem_id = threadIdx.x + blockIdx.x * blockDim.x;                           
  int quad_id = threadIdx.y;                                                     
  if (elem_id < num_elements && quad_id < num_quadrature_points) {               
    destination(elem_id, quad_id) = source(elem_id, quad_id);                    
  }                                                                              
}                                                                                
                                                                                 
int main() {                                                                     
  constexpr int num_elements = 256;                                              
  constexpr int elements_per_block = 16;                                         
  constexpr int num_quadrature_points = 27;                                      
  QuadratureData<int> source(num_elements, num_quadrature_points);               
  QuadratureData<int> destination(num_elements, num_quadrature_points);          
                                                                                 
  dim3 blocks{elements_per_block, num_quadrature_points};                        
  dim3 grids{num_elements / elements_per_block};                                 
                                                                                 
  fill<<<grids, blocks>>>(source, num_elements, num_quadrature_points);          
  copy<<<grids, blocks>>>(destination, source, num_elements, num_quadrature_points);
  cudaDeviceSynchronize();                                                       
                                                                                 
  for (int i = 0; i < 30; i++) {                                                 
    std::cout << source.ptr[i] << " " << destination.ptr[i] << std::endl;        
  }                                                                              
                                                                                 
  source.destroy();                                                              
  destination.destroy();                                                         
}  
