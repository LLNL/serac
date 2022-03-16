#include "sum_factorization.hpp"
#include "sum_factorization_external_cache_tmp.hpp"

using namespace serac;
using std::sin, std::abs;

template <typename trial_space, Geometry geom, int q>
__global__ void preprocess_kernel_with_cache(mfem::DeviceTensor<4, const double> input,
                                             mfem::DeviceTensor<5, double>       output)
{
  static constexpr int  n    = trial_space::order + 1;
  static constexpr auto rule = GaussLegendreRule<geom, q>();

  __shared__ tensor<double, n, n, n> X;
  __shared__ tensor<double, 3, n, n, q> A1;
  __shared__ tensor<double, 3, n, q, q> A2;

  __shared__ tensor<double, q, n> B;
  __shared__ tensor<double, q, n> G;

  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    for (int i = 0; i < q; i++) {
      B[i] = GaussLobattoInterpolation<n>(rule.point(i));
      G[i] = GaussLobattoInterpolationDerivative<n>(rule.point(i));
    }
  }

  for (int dz = threadIdx.z; dz < n; dz += blockDim.z) {
    for (int dy = threadIdx.y; dy < n; dy += blockDim.y) {
      for (int dx = threadIdx.x; dx < n; dx += blockDim.x) {
        X(dz, dy, dx) = input(dx, dy, dz, 0);
      }
    }
  }
  __syncthreads();

  auto qf_input = BatchPreprocessCUDA<trial_space>(X, rule, B, G, A1, A2);

  output(threadIdx.x, threadIdx.y, threadIdx.z, 0, 0) = qf_input.value;
  output(threadIdx.x, threadIdx.y, threadIdx.z, 1, 0) = qf_input.gradient[0];
  output(threadIdx.x, threadIdx.y, threadIdx.z, 2, 0) = qf_input.gradient[1];
  output(threadIdx.x, threadIdx.y, threadIdx.z, 3, 0) = qf_input.gradient[2];
}

template <typename trial_space, Geometry geom, int q>
__global__ void preprocess_kernel(mfem::DeviceTensor<4, const double> input, mfem::DeviceTensor<5, double> output)
{
  static constexpr auto rule     = GaussLegendreRule<geom, q>();
  auto                  qf_input = BatchPreprocessCUDA<trial_space>(input, rule, 0);

  output(threadIdx.x, threadIdx.y, threadIdx.z, 0, 0) = qf_input.value;
  output(threadIdx.x, threadIdx.y, threadIdx.z, 1, 0) = qf_input.gradient[0];
  output(threadIdx.x, threadIdx.y, threadIdx.z, 2, 0) = qf_input.gradient[1];
  output(threadIdx.x, threadIdx.y, threadIdx.z, 3, 0) = qf_input.gradient[2];
}

template <typename trial_space, Geometry geom, int q>
__global__ void postprocess_kernel(mfem::DeviceTensor<4, double> r_e)
{
  static constexpr auto rule = GaussLegendreRule<geom, q>();
  tuple                 qf_output{1.0, tensor<double, 3>{threadIdx.x * 2.0, threadIdx.y * 3.0, threadIdx.z * 5.0}};
  BatchPostprocessCUDA<trial_space>(qf_output, rule, r_e, 0);
}

template <typename trial_space, Geometry geom, int q>
__global__ void postprocess_kernel_with_cache(mfem::DeviceTensor<4, double> r_e)
{
  static constexpr int  n    = trial_space::order + 1;
  static constexpr auto rule = GaussLegendreRule<geom, q>();

  __shared__ tensor<double, 4, q, q, q> f;
  __shared__ tensor<double, 4, q, q, n> A1;
  __shared__ tensor<double, 4, q, n, n> A2;

  __shared__ tensor<double, q, n> B;
  __shared__ tensor<double, q, n> G;

  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    for (int i = 0; i < q; i++) {
      B[i] = GaussLobattoInterpolation<n>(rule.point(i));
      G[i] = GaussLobattoInterpolationDerivative<n>(rule.point(i));
    }
  }

  f(0, threadIdx.z, threadIdx.y, threadIdx.x) = 1.0;
  f(1, threadIdx.z, threadIdx.y, threadIdx.x) = threadIdx.x * 2.0;
  f(2, threadIdx.z, threadIdx.y, threadIdx.x) = threadIdx.y * 3.0;
  f(3, threadIdx.z, threadIdx.y, threadIdx.x) = threadIdx.z * 5.0;
  __syncthreads();

  BatchPostprocessCUDA<trial_space>(f, rule, r_e, 0, B, G, A1, A2);
}

int main()
{
  using serac::Geometry;
  using serac::H1;

  mfem::Device device("cuda");

  constexpr int num_elements = 1;

  {
    constexpr int p = 3;
    constexpr int n = p + 1;
    constexpr int q = 4;

    using test = H1<p>;

    mfem::Vector U1D(num_elements * n * n * n);
    U1D.UseDevice(true);

    auto U = mfem::Reshape(U1D.HostReadWrite(), n, n, n, num_elements);

    for (int ix = 0; ix < n; ix++) {
      for (int iy = 0; iy < n; iy++) {
        for (int iz = 0; iz < n; iz++) {
          U(ix, iy, iz, 0) = 2 * ix - 3.0 * iy + std::sin(iz);
        }
      }
    }

    // clang-format off
    double answers[q][q][q][4]{
      {
        {
          {-0.04622219477163936, 8.338906306602619, -12.50835945990393, 3.590294810165319},
          {0.6197790204557677, 8.338906306602617, -12.50835945990392, 1.517854984151259},
          {0.6717203562044647, 8.338906306602620, -12.50835945990393, -1.218394736107182},
          {0.07823479958972807, 8.338906306602617, -12.50835945990393, -3.340338635918283}
        }, {
          {-2.503329140426160, 8.338906306602615, -7.128637731167952, 3.590294810165319},
          {-1.837327925198753, 8.338906306602613, -7.128637731167952, 1.517854984151259},
          {-1.785386589450056, 8.338906306602613, -7.128637731167951, -1.218394736107183},
          {-2.378872146064792, 8.338906306602611, -7.128637731167951, -3.340338635918279}
        }, {
          {-4.701776618068474, 8.338906306602619, -7.128637731167953, 3.590294810165318},
          {-4.035775402841066, 8.338906306602613, -7.128637731167954, 1.517854984151260},
          {-3.983834067092370, 8.338906306602619, -7.128637731167955, -1.218394736107186},
          {-4.577319623707106, 8.338906306602620, -7.128637731167952, -3.340338635918277}
        }, {
          {-7.158883563722995, 8.338906306602620, -12.50835945990392, 3.590294810165315},
          {-6.492882348495587, 8.338906306602620, -12.50835945990392, 1.517854984151257},
          {-6.440941012746892, 8.338906306602617, -12.50835945990393, -1.218394736107184},
          {-7.034426569361627, 8.338906306602622, -12.50835945990393, -3.340338635918276}
        }
      }, {
        {
          {1.591849102331375, 4.752425154111968, -12.50835945990393, 3.590294810165321},
          {2.257850317558781, 4.752425154111966, -12.50835945990393, 1.517854984151259},
          {2.309791653307478, 4.752425154111968, -12.50835945990393, -1.218394736107181},
          {1.716306096692742, 4.752425154111967, -12.50835945990393, -3.340338635918283}
        }, {
          {-0.8652578433231461, 4.752425154111968, -7.128637731167951, 3.590294810165317},
          {-0.1992566280957394, 4.752425154111966, -7.128637731167949, 1.517854984151258},
          {-0.1473152923470425, 4.752425154111968, -7.128637731167952, -1.218394736107182},
          {-0.7408008489617789, 4.752425154111967, -7.128637731167951, -3.340338635918281}
        }, {
          {-3.063705320965459, 4.752425154111967, -7.128637731167951, 3.590294810165317},
          {-2.397704105738053, 4.752425154111966, -7.128637731167951, 1.517854984151258},
          {-2.345762769989355, 4.752425154111966, -7.128637731167951, -1.218394736107184},
          {-2.939248326604091, 4.752425154111965, -7.128637731167949, -3.340338635918280}
        }, {
          {-5.520812266619981, 4.752425154111966, -12.50835945990392, 3.590294810165321},
          {-4.854811051392573, 4.752425154111966, -12.50835945990392, 1.517854984151261},
          {-4.802869715643876, 4.752425154111967, -12.50835945990392, -1.218394736107183},
          {-5.396355272258612, 4.752425154111966, -12.50835945990392, -3.340338635918281}
        }
      }, {
        {
          {3.057480754092917, 4.752425154111970, -12.50835945990393, 3.590294810165323},
          {3.723481969320323, 4.752425154111969, -12.50835945990393, 1.517854984151259},
          {3.775423305069022, 4.752425154111971, -12.50835945990393, -1.218394736107181},
          {3.181937748454284, 4.752425154111970, -12.50835945990393, -3.340338635918283}
        }, {
          {0.6003738084383958, 4.752425154111968, -7.128637731167951, 3.590294810165320},
          {1.266375023665802, 4.752425154111968, -7.128637731167952, 1.517854984151259},
          {1.318316359414500, 4.752425154111968, -7.128637731167952, -1.218394736107182},
          {0.7248308027997628, 4.752425154111967, -7.128637731167951, -3.340338635918283}
        }, {
          {-1.598073669203918, 4.752425154111967, -7.128637731167952, 3.590294810165318},
          {-0.9320724539765108, 4.752425154111966, -7.128637731167951, 1.517854984151259},
          {-0.8801311182278135, 4.752425154111967, -7.128637731167951, -1.218394736107183},
          {-1.473616674842550, 4.752425154111966, -7.128637731167951, -3.340338635918283}
        }, {
          {-4.055180614858439, 4.752425154111966, -12.50835945990393, 3.590294810165316},
          {-3.389179399631032, 4.752425154111963, -12.50835945990392, 1.517854984151260},
          {-3.337238063882335, 4.752425154111966, -12.50835945990393, -1.218394736107184},
          {-3.930723620497071, 4.752425154111965, -12.50835945990393, -3.340338635918279}
        }
      }, {
        {
          {4.695552051195931, 8.338906306602619, -12.50835945990393, 3.590294810165322},
          {5.361553266423337, 8.338906306602620, -12.50835945990392, 1.517854984151259},
          {5.413494602172036, 8.338906306602613, -12.50835945990392, -1.218394736107180},
          {4.820009045557298, 8.338906306602613, -12.50835945990393, -3.340338635918285}
        }, {
          {2.238445105541410, 8.338906306602613, -7.128637731167951, 3.590294810165319},
          {2.904446320768816, 8.338906306602610, -7.128637731167951, 1.517854984151259},
          {2.956387656517514, 8.338906306602613, -7.128637731167951, -1.218394736107181},
          {2.362902099902776, 8.338906306602615, -7.128637731167951, -3.340338635918283}
        }, {
          {0.03999762789909700, 8.338906306602617, -7.128637731167951, 3.590294810165320},
          {0.7059988431265038, 8.338906306602619, -7.128637731167950, 1.517854984151259},
          {0.7579401788752009, 8.338906306602619, -7.128637731167951, -1.218394736107181},
          {0.1644546222604643, 8.338906306602615, -7.128637731167951, -3.340338635918283}
        }, {
          {-2.417109317755424, 8.338906306602617, -12.50835945990392, 3.590294810165317},
          {-1.751108102528017, 8.338906306602615, -12.50835945990392, 1.517854984151258},
          {-1.699166766779320, 8.338906306602622, -12.50835945990393, -1.218394736107183},
          {-2.292652323394057, 8.338906306602624, -12.50835945990392, -3.340338635918281}
        }
      }
    };
    // clang-format on

    dim3 blocksize{q, q, q};
    int  gridsize = num_elements;

    {
      mfem::Vector R1D(num_elements * 4 * q * q * q);
      R1D                                     = 0.0;
      mfem::DeviceTensor<4, const double> u_d = mfem::Reshape(U1D.Read(), n, n, n, num_elements);
      mfem::DeviceTensor<5, double>       r_d = mfem::Reshape(R1D.ReadWrite(), q, q, q, 4, num_elements);
      preprocess_kernel<test, Geometry::Hexahedron, q><<<gridsize, blocksize>>>(u_d, r_d);
      cudaDeviceSynchronize();

      mfem::DeviceTensor<5, const double> r_h = mfem::Reshape(R1D.HostRead(), q, q, q, 4, num_elements);

      for (int i = 0; i < q; i++) {
        for (int j = 0; j < q; j++) {
          for (int k = 0; k < q; k++) {
            for (int c = 0; c < 4; c++) {
              auto relative_error = abs(r_h(i, j, k, c, 0) - answers[i][j][k][c]) / abs(answers[i][j][k][c]);
              if (relative_error > 5.0e-14) {
                std::cout << "error: " << r_h(i, j, k, c, 0) << " " << answers[i][j][k][c] << ", " << relative_error
                          << std::endl;
              }
            }
          }
        }
      }
    }

    {
      mfem::Vector R1D(num_elements * 4 * q * q * q);
      R1D                                     = 0.0;
      mfem::DeviceTensor<4, const double> u_d = mfem::Reshape(U1D.Read(), n, n, n, num_elements);
      mfem::DeviceTensor<5, double>       r_d = mfem::Reshape(R1D.ReadWrite(), q, q, q, 4, num_elements);
      preprocess_kernel_with_cache<test, Geometry::Hexahedron, q><<<gridsize, blocksize>>>(u_d, r_d);
      cudaDeviceSynchronize();

      mfem::DeviceTensor<5, const double> r_h = mfem::Reshape(R1D.HostRead(), q, q, q, 4, num_elements);

      for (int i = 0; i < q; i++) {
        for (int j = 0; j < q; j++) {
          for (int k = 0; k < q; k++) {
            for (int c = 0; c < 4; c++) {
              auto relative_error = abs(r_h(i, j, k, c, 0) - answers[i][j][k][c]) / abs(answers[i][j][k][c]);
              if (relative_error > 5.0e-14) {
                std::cout << "error: " << r_h(i, j, k, c, 0) << " " << answers[i][j][k][c] << ", " << relative_error
                          << std::endl;
              }
            }
          }
        }
      }
    }
  }

  {
    constexpr int p = 3;
    constexpr int n = p + 1;
    constexpr int q = 3;

    using test = H1<p>;

    mfem::Vector U1D(num_elements * n * n * n);
    mfem::Vector R1D(num_elements * 4 * q * q * q);
    U1D.UseDevice(true);
    R1D.UseDevice(true);

    auto U = mfem::Reshape(U1D.HostReadWrite(), n, n, n, num_elements);

    for (int ix = 0; ix < n; ix++) {
      for (int iy = 0; iy < n; iy++) {
        for (int iz = 0; iz < n; iz++) {
          U(ix, iy, iz, 0) = 2 * ix - 3.0 * iy + sin(iz);
        }
      }
    }

    R1D = 0.0;

    mfem::DeviceTensor<4, const double> u_d = mfem::Reshape(U1D.Read(), n, n, n, num_elements);
    mfem::DeviceTensor<5, double>       r_d = mfem::Reshape(R1D.ReadWrite(), q, q, q, 4, num_elements);

    dim3 blocksize{q, q, q};
    int  gridsize = num_elements;
    preprocess_kernel<test, Geometry::Hexahedron, q><<<gridsize, blocksize>>>(u_d, r_d);
    cudaDeviceSynchronize();

    mfem::DeviceTensor<5, const double> r_h = mfem::Reshape(R1D.HostRead(), q, q, q, 4, num_elements);

    // clang-format off
    double answers[q][q][q][4]{
      {
        {
          {-0.06976517422279016,7.527864045000420,-11.29179606750063,3.247646211241188},
          {0.5905504600868352,7.527864045000422,-11.29179606750063,0.1543006667646776},
          {0.04362979171617654,7.527864045000420,-11.29179606750063,-2.986495249049149}
        }, {
          {-3.111645785692786,7.527864045000416,-6.135254915624209,3.247646211241187},
          {-2.451330151383162,7.527864045000420,-6.135254915624211,0.1543006667646788},
          {-2.998250819753819,7.527864045000419,-6.135254915624211,-2.986495249049150}
        }, {
          {-6.153526397162780,7.527864045000409,-11.29179606750063,3.247646211241181},
          {-5.493210762853155,7.527864045000412,-11.29179606750063,0.1543006667646789},
          {-6.040131431223813,7.527864045000410,-11.29179606750063,-2.986495249049152}
        }
      }, { 
        {
          {1.958155233423875,4.090169943749476,-11.29179606750063,3.247646211241189},
          {2.618470867733500,4.090169943749474,-11.29179606750063,0.1543006667646782},
          {2.071550199362840,4.090169943749473,-11.29179606750063,-2.986495249049149}
        }, {
          {-1.083725378046122,4.090169943749474,-6.135254915624211,3.247646211241187},
          {-0.4234097437364975,4.090169943749475,-6.135254915624213,0.1543006667646782},
          {-0.9703304121071556,4.090169943749474,-6.135254915624212,-2.986495249049151}
        }, {
          {-4.125605989516117,4.090169943749475,-11.29179606750063,3.247646211241185},
          {-3.465290355206493,4.090169943749473,-11.29179606750063,0.1543006667646784},
          {-4.012211023577151,4.090169943749473,-11.29179606750063,-2.986495249049152}
        }
      }, {
        {
          {3.986075641070537,7.527864045000417,-11.29179606750063,3.247646211241192},
          {4.646391275380163,7.527864045000422,-11.29179606750063,0.1543006667646779},
          {4.099470607009503,7.527864045000422,-11.29179606750063,-2.986495249049149}
        }, {
          {0.9441950296005416,7.527864045000419,-6.135254915624211,3.247646211241188},
          {1.604510663910167,7.527864045000424,-6.135254915624213,0.1543006667646776},
          {1.057589995539508,7.527864045000420,-6.135254915624209,-2.986495249049149}
        }, {
          {-2.097685581869453,7.527864045000414,-11.29179606750063,3.247646211241186},
          {-1.437369947559828,7.527864045000416,-11.29179606750063,0.1543006667646778},
          {-1.984290615930487,7.527864045000415,-11.29179606750062,-2.986495249049149}
        }
      }
    };
    // clang-format on

    for (int i = 0; i < q; i++) {
      for (int j = 0; j < q; j++) {
        for (int k = 0; k < q; k++) {
          for (int c = 0; c < 4; c++) {
            auto relative_error = abs(r_h(i, j, k, c, 0) - answers[i][j][k][c]) / abs(answers[i][j][k][c]);
            if (relative_error > 5.0e-14) {
              std::cout << "error: " << r_h(i, j, k, c, 0) << " " << answers[i][j][k][c] << ", " << relative_error
                        << std::endl;
            }
          }
        }
      }
    }
  }

  {
    constexpr int p = 3;
    constexpr int n = p + 1;
    constexpr int q = 2;

    using test = H1<p>;

    mfem::Vector R1D(num_elements * n * n * n);
    R1D.UseDevice(true);

    R1D = 0.0;

    mfem::DeviceTensor<4, double> r_d = mfem::Reshape(R1D.ReadWrite(), n, n, n, num_elements);

    dim3 blocksize{q, q, q};
    int  gridsize = num_elements;
    // postprocess_kernel<test, Geometry::Hexahedron, q><<<gridsize, blocksize>>>(r_d);
    postprocess_kernel_with_cache<test, Geometry::Hexahedron, q><<<gridsize, blocksize>>>(r_d);
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
          auto relative_error = abs(r_h(i, j, k, 0) - answers[i][j][k]) / abs(answers[i][j][k]);
          if (relative_error > 5.0e-14) {
            std::cout << "error: " << r_h(i, j, k, 0) << " " << answers[i][j][k] << ", " << relative_error << std::endl;
          }
        }
      }
    }
  }

  {
    constexpr int p = 2;
    constexpr int n = p + 1;
    constexpr int q = 4;

    using test = H1<p>;

    mfem::Vector R1D(num_elements * n * n * n);
    R1D.UseDevice(true);

    R1D = 0.0;

    mfem::DeviceTensor<4, double> r_d = mfem::Reshape(R1D.ReadWrite(), n, n, n, num_elements);

    dim3 blocksize{q, q, q};
    int  gridsize = num_elements;
    // postprocess_kernel<test, Geometry::Hexahedron, q><<<gridsize, blocksize>>>(r_d);
    postprocess_kernel_with_cache<test, Geometry::Hexahedron, q><<<gridsize, blocksize>>>(r_d);
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
          auto relative_error = abs(r_h(i, j, k, 0) - answers[i][j][k]) / abs(answers[i][j][k]);
          if (relative_error > 5.0e-14) {
            std::cout << "error: " << r_h(i, j, k, 0) << " " << answers[i][j][k] << ", " << relative_error << std::endl;
          }
        }
      }
    }
  }

  {
    constexpr int p = 2;
    constexpr int n = p + 1;
    constexpr int q = 3;

    using test = H1<p>;

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

    dim3 blocksize{q, q, q};
    int  gridsize = num_elements;

    {
      mfem::Vector R1D(num_elements * n * n * n);
      R1D = 0.0;

      mfem::DeviceTensor<4, double> r_d = mfem::Reshape(R1D.ReadWrite(), n, n, n, num_elements);
      postprocess_kernel<test, Geometry::Hexahedron, q><<<gridsize, blocksize>>>(r_d);
      cudaDeviceSynchronize();

      mfem::DeviceTensor<4, const double> r_h = mfem::Reshape(R1D.HostRead(), n, n, n, num_elements);

      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
          for (int k = 0; k < n; k++) {
            auto relative_error = abs(r_h(i, j, k, 0) - answers[i][j][k]) / abs(answers[i][j][k]);
            if (relative_error > 5.0e-14) {
              std::cout << "error: " << r_h(i, j, k, 0) << " " << answers[i][j][k] << ", " << relative_error
                        << std::endl;
            }
          }
        }
      }
    }

    {
      mfem::Vector R1D(num_elements * n * n * n);
      R1D = 0.0;

      mfem::DeviceTensor<4, double> r_d = mfem::Reshape(R1D.ReadWrite(), n, n, n, num_elements);
      postprocess_kernel_with_cache<test, Geometry::Hexahedron, q><<<gridsize, blocksize>>>(r_d);
      cudaDeviceSynchronize();

      mfem::DeviceTensor<4, const double> r_h = mfem::Reshape(R1D.HostRead(), n, n, n, num_elements);

      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
          for (int k = 0; k < n; k++) {
            auto relative_error = abs(r_h(i, j, k, 0) - answers[i][j][k]) / abs(answers[i][j][k]);
            if (relative_error > 5.0e-14) {
              std::cout << "error: " << r_h(i, j, k, 0) << " " << answers[i][j][k] << ", " << relative_error
                        << std::endl;
            }
          }
        }
      }
    }

  }
}
