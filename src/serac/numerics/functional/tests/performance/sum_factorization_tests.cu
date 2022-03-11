#include "sum_factorization.hpp"

using namespace serac;

template <typename trial_space, Geometry geom, int q>
__global__ void postprocess_kernel(mfem::DeviceTensor<4, double> r_e)
{
  static constexpr auto rule = GaussLegendreRule<geom, q>();
  tuple                 qf_output{1.0, tensor<double, 3>{threadIdx.x * 2.0, threadIdx.y * 3.0, threadIdx.z * 5.0}};
  BatchPostprocessCUDA<trial_space>(qf_output, rule, r_e, 0);
}

int main()
{
  using serac::Geometry;
  using serac::H1;


  mfem::Device device("cuda");

  constexpr int num_elements = 1;

  {
    constexpr int p            = 3;
    constexpr int n            = p + 1;
    constexpr int q            = 2;

    using test  = H1<p>;

    mfem::Vector R1D(num_elements * n * n * n);
    R1D.UseDevice(true);

    R1D = 0.0;

    mfem::DeviceTensor<4, double> r_d = mfem::Reshape(R1D.ReadWrite(), n, n, n, num_elements);
    dim3                          blocksize{q, q, q};
    int                           gridsize = num_elements;
    postprocess_kernel<test, Geometry::Hexahedron, q><<<gridsize, blocksize>>>(r_d);
    cudaDeviceSynchronize();

    mfem::DeviceTensor<4, const double> r_h = mfem::Reshape(R1D.HostRead(), n, n, n, num_elements);

    // clang-format off
    double answers[n][n][n]{
      {
        {0.1277895387890918, 0.1305790775781840, 0.1305790775781842, 0.4055673165668692},
        {0.3339265241250943, -0.8722104612109071, -0.8722104612109070, 1.722815413013982},
        {0.3339265241250945, -0.8722104612109070, -0.8722104612109070, 1.722815413013983},
        {0.2944562054557583, 0.9639124109115168, 0.9639124109115172, 0.5722339832335358}
      }, {
        {0.4356002473985494, -0.3638418448436311, -0.3638418448436308, 1.824489136287438},
        {0.6528953878909205, -9.444738469727300, -9.444738469727302, 7.597339832335364},
        {0.6528953878909211, -9.444738469727303, -9.444738469727303, 7.597339832335368},
        {1.268933580731882, 3.802824821823035, 3.802824821823037, 2.657822469620771}
      }, {
        {0.4356002473985495, -0.3638418448436310, -0.3638418448436308, 1.824489136287438},
        {0.6528953878909209, -9.444738469727302, -9.444738469727305, 7.597339832335367},
        {0.6528953878909215, -9.444738469727307, -9.444738469727309, 7.597339832335372},
        {1.268933580731883, 3.802824821823036, 3.802824821823039, 2.657822469620772}
      }, {
        {0.2389006499002028, 0.6861346331337391, 0.6861346331337396, 0.5166784276779803},
        {0.8894820796806495, 1.905567316566870, 1.905567316566872, 2.278370968569538},
        {0.8894820796806499, 1.905567316566871, 1.905567316566873, 2.278370968569539},
        {0.4055673165668693, 1.519467966467072, 1.519467966467073, 0.6833450943446471}
      }
    };
    // clang-format on

    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        for (int k = 0; k < n; k++) {
          auto relative_error = abs(r_h(i,j,k,0) - answers[i][j][k]) / abs(answers[i][j][k]);
          if (relative_error > 5.0e-14) {
            std::cout << "error: " << r_h(i,j,k,0) << " " << answers[i][j][k] << ", " << relative_error << std::endl;
          }
        }
      }
    }

  }

  {
    constexpr int p            = 2;
    constexpr int n            = p + 1;
    constexpr int q            = 4;

    using test  = H1<p>;

    mfem::Vector R1D(num_elements * n * n * n);
    R1D.UseDevice(true);

    R1D = 0.0;

    mfem::DeviceTensor<4, double> r_d = mfem::Reshape(R1D.ReadWrite(), n, n, n, num_elements);
    dim3                          blocksize{q, q, q};
    int                           gridsize = num_elements;
    postprocess_kernel<test, Geometry::Hexahedron, q><<<gridsize, blocksize>>>(r_d);
    cudaDeviceSynchronize();

    mfem::DeviceTensor<4, const double> r_h = mfem::Reshape(R1D.HostRead(), n, n, n, num_elements);

    // clang-format off
    double answers[n][n][n]{
      {
        {-0.4959606677266817, -42.77756512053472, 43.58567198533451}, 
        {-26.19556378456263, -180.4015123319811, 91.35545662360063}, 
        {25.95301892411004, 27.75304712436323, 70.03465157717123}
      }, {
        {-17.90456311657658, -158.2921772173516, 99.64645729158666}, 
        {-114.0735069880926, -598.9871534966403, 199.3958807670095}, 
        {52.62604912832137, 29.78945543570973, 170.1770695364846}
      }, {
        {17.13669239349780, 4.242843042730591, 61.21832504655899}, 
        {20.82484437870268, -55.01375722994018, 138.3758647868659}, 
        {43.58567198533453, 74.77345528762854, 87.66730463839572}
      }
    };
    // clang-format on

    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        for (int k = 0; k < n; k++) {
          auto relative_error = abs(r_h(i,j,k,0) - answers[i][j][k]) / abs(answers[i][j][k]);
          if (relative_error > 5.0e-14) {
            std::cout << "error: " << r_h(i,j,k,0) << " " << answers[i][j][k] << ", " << relative_error << std::endl;
          }
        }
      }
    }

  }

  {
    constexpr int p            = 2;
    constexpr int n            = p + 1;
    constexpr int q            = 3;

    using test  = H1<p>;

    mfem::Vector R1D(num_elements * n * n * n);
    R1D.UseDevice(true);

    R1D = 0.0;

    mfem::DeviceTensor<4, double> r_d = mfem::Reshape(R1D.ReadWrite(), n, n, n, num_elements);
    dim3                          blocksize{q, q, q};
    int                           gridsize = num_elements;
    postprocess_kernel<test, Geometry::Hexahedron, q><<<gridsize, blocksize>>>(r_d);
    cudaDeviceSynchronize();

    mfem::DeviceTensor<4, const double> r_h = mfem::Reshape(R1D.HostRead(), n, n, n, num_elements);

    // clang-format off
    double answers[n][n][n]{
      {
        {0.5701920370773611, -9.974903981461324, 11.37019203707736}, 
        {-5.300711944383957, -50.95857611123209, 27.09928805561605}, 
        {7.050192037077360, 9.465096018538679, 17.85019203707736}
      }, {
        {-2.963615925845279, -43.94728805561604, 29.43638407415473}, 
        {-29.92471194438396, -194.9434566673925, 67.27528805561603}, 
        {16.47638407415472, 14.37271194438396, 48.87638407415472}
      }, {
        {4.890192037077362, 2.985096018538675, 15.69019203707736}, 
        {7.659288055616042, -12.07857611123210, 40.05928805561604}, 
        {11.37019203707736, 22.42509601853867, 22.17019203707736}
      }
    };
    // clang-format on

    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        for (int k = 0; k < n; k++) {
          auto relative_error = abs(r_h(i,j,k,0) - answers[i][j][k]) / abs(answers[i][j][k]);
          if (relative_error > 5.0e-14) {
            std::cout << "error: " << r_h(i,j,k,0) << " " << answers[i][j][k] << ", " << relative_error << std::endl;
          }
        }
      }
    }

  }


}
