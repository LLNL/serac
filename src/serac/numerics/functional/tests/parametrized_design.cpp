#include "serac/mesh/mesh_utils_base.hpp"
#include "serac/numerics/functional/domain.hpp"
#include "serac/numerics/functional/functional.hpp"

using namespace serac;

using vec2 = tensor<double,2>;

namespace impl {

  template < typename T, int m >
  tensor<T, m> abs(const tensor<T,m> & x) {
    tensor<T,m> abs_x = x;
    for (int i = 0; i < m; i++) {
      if (abs_x[i] < 0.0) {
        abs_x[i] = -1.0 * abs_x[i];
      }
    }
    return abs_x;
  }

  double max(double x, double y) {
    return (x < y) ? y : x;
  }

  template < typename T >
  dual<T> max(const dual<T> & x, const dual<T> & y) {
    return (x < y) ? y : x;
  }

  double min(double x, double y) {
    return (x < y) ? x : y;
  }

  template < typename T >
  dual<T> min(const dual<T> & x, const dual<T> & y) {
    return (x < y) ? x : y;
  }

  template < typename T, int m >
  T max(const tensor<T,m> & x) {
    T max_x = x[0];
    for (int i = 1; i < m; i++) {
      if (x[i] > max_x) {
        max_x = x[i];
      }
    }
    return max_x;
  }

  template < typename T, int m >
  tensor<T, m> max(const tensor<T,m> & x, T y) {
    tensor<T,m> max_x{};
    for (int i = 0; i < m; i++) {
      max_x[i] = (x[i] > y) ?  x[i] : y;
    }
    return max_x;
  }

  template < typename T, int m >
  T norm(const tensor<T,m> & v) {
    using std::sqrt;
    T sq_sum{};
    for (int i = 0; i < m; i++) {
      sq_sum += v[i] * v[i];
    }
    sq_sum = max(sq_sum, 1.0e-15); // avoid exact zeros
    return sqrt(sq_sum);
  }
}

template < typename T, std::size_t n >
T min(std::array< T, n > values) {
  static_assert(n >= 1);
  T min_value = values[0];
  for (std::size_t i = 1; i < n; i++) {
    min_value = impl::min(min_value, values[i]); 
  }
  return min_value;
}

template < typename T, std::size_t n >
T smooth_min(std::array< T, n > values, double r) {
  T weighted_sum{};
  for (auto x : values) {
    weighted_sum += exp(-x / r);
  }
  return -r * log(weighted_sum);
}

template < typename T >
T sdf_difference(T a, T b) {
  return impl::max(a, -b);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

struct Disk {
  vec2 c;
  double r;

  template < typename T >
  T SDF(const tensor<T, 2> & p) const {
    return norm(p - c) - r;
  };
};

struct Box { 
  vec2 min;
  vec2 max; 

  template < typename T >
  T SDF(const tensor<T, 2> & p) const {
    constexpr T zero{};
    auto center = (min + max) * 0.5;
    auto halfwidths = (max - min) * 0.5;
    auto q = impl::abs(p - center) - halfwidths;
    return impl::norm(impl::max(q, zero)) + impl::min(impl::max(q), zero);
  }
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template < typename T >
auto mu(const tensor< T, 2, 2 > & J) {
  return 0.5 * (inner(J, J) / det(J)) - 1.0;
}

template < typename T >
T phi(const tensor<T, 2> & x, std::array< double, 8 > p) {
  auto [x1, r1, t1, x2, r2, t2, h, rho] = p;

  Disk left_disk{{x1, 0.0}, r1 + t1};
  Box handle{{x1, -0.5 * h}, {x2, 0.5 * h}};
  Disk right_disk{{x2, 0.0}, r2 + t2};

  Disk left_hole{{x1, 0.0}, r1};
  Box left_slot{{x1 - 1.5 * (r1 + t1), -r1}, {x1, r1}};
  Disk right_hole{{x2, 0.0}, r2};

  T SDF1 = smooth_min(std::array{left_disk.SDF(x), handle.SDF(x), right_disk.SDF(x)}, rho);
  T SDF2 = min(std::array{left_hole.SDF(x), left_slot.SDF(x), right_hole.SDF(x)});

  return sdf_difference(SDF1, SDF2);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[])
{
  int num_procs, myid;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  axom::slic::SimpleLogger logger;

  int serial_refinement = 0;
  int parallel_refinement = 0;
  std::string meshfile = SERAC_REPO_DIR "/data/meshes/parameterized.msh";
  std::unique_ptr<mfem::ParMesh> mesh = mesh::refineAndDistribute(buildMeshFromFile(meshfile), serial_refinement, parallel_refinement);

  // these are the values of the parameters used to generate parameterized.msh
  std::array< double, 8 > parameters = {
    -1.00,  // x1
     0.15,  // r1
     0.10,  // t1
     0.20,  // x2
     0.125, // r2
     0.10,  // t2
     0.2,   // h
     0.05   // fillet "radius"
  };

  // make one of the parameters different than what was used to generate the mesh
  parameters[0] *= 1.3;

  using trial_space = H1<1, 2>;

  auto [fes, fec] = generateParFiniteElementSpace<trial_space>(mesh.get());

  serac::Functional< double(trial_space) > F({fes.get()});

  F.AddDomainIntegral(Dimension<2>{}, DependsOn<0>{}, [](double t, auto X, auto u){
    auto dX_dxi = get<1>(X);
    auto du_dX = get<1>(u);
    auto dx_dxi = dX_dxi + dot(du_dX, dX_dxi);
    return (1.0 - t) * mu(dx_dxi);
  }, *mesh);

  F.AddBoundaryIntegral(Dimension<1>{}, DependsOn<0>{}, [&parameters](double t, auto X, auto u){
    auto x = get<0>(X) + get<0>(u);
    auto phi_value = phi(x, parameters);
    return t * phi_value * phi_value;
  }, *mesh);

  mfem::ParGridFunction u_gf(fes.get());
  std::unique_ptr<mfem::HypreParVector> u(fes->NewTrueDofVector());
  //u->Randomize(0);
  *u = 0;

  mfem::ParaViewDataCollection paraview_dc("parametrized_design", mesh.get());
  paraview_dc.SetPrefixPath("ParaView");
  paraview_dc.SetLevelsOfDetail(1);
  paraview_dc.SetDataFormat(mfem::VTKFormat::BINARY);
  paraview_dc.RegisterField("displacement", &u_gf);

  for (int k = 0; k < 100; k++) {

    auto [Fvalue, dF_du] = F(1.0, differentiate_wrt(*u));

    auto dF_du_vector = assemble(dF_du);

    std::cout << k << " " << Fvalue << std::endl;
    
    u->Add(-5.0, *dF_du_vector);

    u_gf.SetFromTrueDofs(*u);

    paraview_dc.SetCycle(k);
    paraview_dc.SetTime(k);
    paraview_dc.Save();
  }

  MPI_Finalize();

  return 0;
}
