namespace serac {

template <typename trial_space, int n, Geometry geom, int q>
__device__ auto BatchPreprocessCUDA(const tensor<double, n, n, n>& X, GaussLegendreRule<geom, q> rule,
                                    const tensor<double, q, n>& B, const tensor<double, q, n>& G,
                                    tensor<double, 3, n, n, q>& A1, tensor<double, 3, n, n, q>& A2)
{

  if constexpr (geom == Geometry::Hexahedron) {
    for (int dz = threadIdx.z; dz < n; dz += blockDim.z) {
      for (int dy = threadIdx.y; dy < n; dy += blockDim.y) {
        for (int qx = threadIdx.x; qx < q; qx += blockDim.x) {
          double sum[2]{};
          for (int dx = 0; dx < n; dx++) {
            sum[0] += B(qx, dx) * X(dz, dy, dx);
            sum[1] += G(qx, dx) * X(dz, dy, dx);
          }
          A1(0, dz, dy, qx) = sum[0];
          A1(1, dz, dy, qx) = sum[1];
        }
      }
    }
    __syncthreads();

    for (int dz = threadIdx.z; dz < n; dz += blockDim.z) {
      for (int qy = threadIdx.y; qy < q; qy += blockDim.y) {
        for (int qx = threadIdx.x; qx < q; qx += blockDim.x) {
          double sum[3]{};
          for (int dy = 0; dy < n; dy++) {
            sum[0] += B(qy, dy) * A1(0, dz, dy, qx);
            sum[1] += B(qy, dy) * A1(1, dz, dy, qx);
            sum[2] += G(qy, dy) * A1(0, dz, dy, qx);
          }
          A2(0, dz, qy, qx) = sum[0];
          A2(1, dz, qy, qx) = sum[1];
          A2(2, dz, qy, qx) = sum[2];
        }
      }
    }
    __syncthreads();

    value_and_gradient<double, tensor<double, 3> > qf_input{};

    for (int qz = threadIdx.z; qz < q; qz += blockDim.z) {
      for (int qy = threadIdx.y; qy < q; qy += blockDim.y) {
        for (int qx = threadIdx.x; qx < q; qx += blockDim.x) {
          for (int dz = 0; dz <n; dz++) {
            qf_input.value       += B(qz, dz) * A2(0, dz, qy, qx);
            qf_input.gradient[0] += B(qz, dz) * A2(1, dz, qy, qx);
            qf_input.gradient[1] += B(qz, dz) * A2(2, dz, qy, qx);
            qf_input.gradient[2] += G(qz, dz) * A2(0, dz, qy, qx);
          }
        }
      }
    }

    return qf_input;
  }
}

template <typename trial_space, Geometry geom, int q, int n>
__device__ void BatchPostprocessCUDA(const tensor<double, 4, q, q, q> & f, GaussLegendreRule<geom, q>, mfem::DeviceTensor<4, double> r_e, int e,
                                     const tensor<double, q, n>& B, const tensor<double, q, n>& G,
                                     tensor<double, 4, q, q, n>& A1, tensor<double, 4, q, n, n>& A2) {

  if constexpr (geom == Geometry::Hexahedron) {

    for (int qz = threadIdx.z; qz < q; qz += blockDim.z) {
      for (int qy = threadIdx.y; qy < q; qy += blockDim.y) {
        for (int dx = threadIdx.x; dx < n; dx += blockDim.x) {
          double sum[4]{};
          for (int qx = 0; qx < q; qx++) {
            sum[0] += B(qx, dx) * f(0, qz, qy, qx);
            sum[1] += G(qx, dx) * f(1, qz, qy, qx);
            sum[2] += B(qx, dx) * f(2, qz, qy, qx);
            sum[3] += B(qx, dx) * f(3, qz, qy, qx);
          }
          A1(0, qz, qy, dx) = sum[0];
          A1(1, qz, qy, dx) = sum[1];
          A1(2, qz, qy, dx) = sum[2];
          A1(3, qz, qy, dx) = sum[3];
        }
      }
    }
    __syncthreads();

    for (int qz = threadIdx.z; qz < q; qz += blockDim.z) {
      for (int dy = threadIdx.y; dy < n; dy += blockDim.y) {
        for (int dx = threadIdx.x; dx < n; dx += blockDim.x) {
          double sum[4]{};
          for (int qy = 0; qy < q; qy++) {
            sum[0] += B(qy, dy) * A1(0, qz, qy, dx);
            sum[1] += B(qy, dy) * A1(1, qz, qy, dx);
            sum[2] += G(qy, dy) * A1(2, qz, qy, dx);
            sum[3] += B(qy, dy) * A1(3, qz, qy, dx);
          }
          A2(0, qz, dy, dx) = sum[0];
          A2(1, qz, dy, dx) = sum[1];
          A2(2, qz, dy, dx) = sum[2];
          A2(3, qz, dy, dx) = sum[3];
        }
      }
    }
    __syncthreads();

    for (int dz = threadIdx.z; dz < n; dz += blockDim.z) {
      for (int dy = threadIdx.y; dy < n; dy += blockDim.y) {
        for (int dx = threadIdx.x; dx < n; dx += blockDim.x) {
          double sum[4]{};
          for (int qz = 0; qz < q; qz++) {
            sum[0] += B(qz, dz) * A2(0, qz, dy, dx);
            sum[1] += B(qz, dz) * A2(1, qz, dy, dx);
            sum[2] += B(qz, dz) * A2(2, qz, dy, dx);
            sum[3] += G(qz, dz) * A2(3, qz, dy, dx);
          }
          r_e(dx,dy,dz,e) += sum[0] + sum[1] + sum[2] + sum[3];
        }
      }
    }

  }

}

}  // namespace serac
