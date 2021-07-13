// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/serac_config.hpp"
#include "serac/numerics/mesh_utils_base.hpp"

#include "serac/physics/utilities/quadrature_data.hpp"

using serac::QuadratureData;

// template <typename T>
// struct QuadratureData {
//   QuadratureData(mfem::Mesh& mesh) : stride(nquad)
//   {
//     const int nelem = mesh.GetNE();
//     // FIXME: This assumes a homogeneous mesh
//     const int geom  = mesh.GetElementBaseGeometry(0);
//     const int nquad = mfem::IntRules.Get(geom, p).GetNPoints();
//     cudaMallocManaged(&ptr, sizeof(T) * nelem * nquad);
//   };

//   void destroy() { cudaFree(ptr); }

//   __host__ __device__ T& operator()(int e, int q) { return ptr[e * stride + q]; }
//   T*                     ptr;
//   int                    stride;
// };

template <typename T>
__global__ void fill(QuadratureData<T>& output, int num_elements, int num_quadrature_points)
{
  int elem_id = threadIdx.x + blockIdx.x * blockDim.x;
  int quad_id = threadIdx.y;
  if (elem_id < num_elements && quad_id < num_quadrature_points) {
    output(elem_id, quad_id) = elem_id * elem_id + quad_id;
  }
}

template <typename T>
__global__ void copy(QuadratureData<T>& destination, QuadratureData<T>& source, int num_elements,
                     int num_quadrature_points)
{
  int elem_id = threadIdx.x + blockIdx.x * blockDim.x;
  int quad_id = threadIdx.y;
  if (elem_id < num_elements && quad_id < num_quadrature_points) {
    destination(elem_id, quad_id) = source(elem_id, quad_id);
  }
}

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);

  axom::slic::SimpleLogger logger;  // create & initialize test logger, finalized when exiting main scope

  constexpr auto mesh_file          = SERAC_REPO_DIR "/data/meshes/star.mesh";
  auto           mesh               = serac::mesh::refineAndDistribute(serac::buildMeshFromFile(mesh_file), 0, 0);
  constexpr int  p                  = 1;
  constexpr int  elements_per_block = 16;
  const int      num_elements       = mesh->GetNE();

  // FIXME: This assumes a homogeneous mesh
  const int geom                  = mesh->GetElementBaseGeometry(0);
  const int num_quadrature_points = mfem::IntRules.Get(geom, p).GetNPoints();

  QuadratureData<int> source(*mesh, p);
  QuadratureData<int> destination(*mesh, p);

  dim3 blocks{elements_per_block, static_cast<unsigned int>(num_quadrature_points)};
  dim3 grids{static_cast<unsigned int>(num_elements / elements_per_block)};

  fill<<<grids, blocks>>>(source, num_elements, num_quadrature_points);
  copy<<<grids, blocks>>>(destination, source, num_elements, num_quadrature_points);
  cudaDeviceSynchronize();

  // Trick with std::transform to get what's roughly a zip iterator
  std::vector<int> temp;
  std::transform(source.begin(), source.end(), destination.begin(), std::back_inserter(temp),
                 [](const auto& src, const auto& dest) {
                   std::cout << src << " " << dest << std::endl;
                   return 0;
                 });

  MPI_Finalize();
}
