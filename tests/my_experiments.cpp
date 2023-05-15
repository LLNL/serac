#include "gtest/gtest.h"

//#include "tuple.hpp"
#include "tensor.hpp"
#include "tuple_tensor_dual_functions.hpp"
#include "materials.hpp"

int enzyme_dup;
int enzyme_out;
int enzyme_const;

template < typename return_type, typename ... T >
extern return_type __enzyme_fwddiff(void*, T ... );


namespace serac {

void NeoHookean_wrapper(const NeoHookean& material, NeoHookean::State& state, const tensor<double, 3, 3>& du_dX, double K, double G,
                        tensor<double, 3, 3>& stress)
{
    stress = material(state, du_dX, K, G);
}

tensor<double, 3, 3> compute_stress(const NeoHookean& material, NeoHookean:: State& state, const tensor<double, 3, 3>& du_dX, double K, double G)
{
    tensor<double, 3, 3> stress{};
    NeoHookean_wrapper(material, state, du_dX, K, G, stress);
    return stress;
}

void J2_wrapper(const J2& material, J2::State& state, const tensor<double, 3, 3>& du_dX, double E, double nu,
    double sigma_y, double Hi, double Hk, tensor<double, 3, 3>& stress)
{
    stress = material(state, du_dX, E, nu, sigma_y, Hi, Hk);
}

}


TEST(material, CanBeCalled) {
    serac::tensor<double, 3, 3> H{{{0.19165388147925, 0.44595621086207, 0.03873204915048},
                                   {0.58923368584434, 0.0923605872371 , 0.25974694007571},
                                   {0.83097065578267, 0.48547287595839, 0.03308538443643}}};
    serac::NeoHookean mat{.density=1.0};
    serac::NeoHookean::State state;
    auto stress = mat(state, H, 1.0, 1.0);
    EXPECT_NE(stress[0][0], 0.0);
}

TEST(material, ParameterDerivativeWithEmptyState) {
    serac::NeoHookean mat{.density=1.0};
    serac::NeoHookean::State state;
    serac::tensor<double, 3, 3> H{{{0.19165388147925, 0.44595621086207, 0.03873204915048},
                                   {0.58923368584434, 0.0923605872371 , 0.25974694007571},
                                   {0.83097065578267, 0.48547287595839, 0.03308538443643}}};
    double K = 1.0;
    double G = 1.0;
    
    // tangent values
    serac::NeoHookean::State dstate;
    serac::tensor<double, 3, 3> dH{};
    double dK = 0.0;
    double dG = 1.0;

    auto dstress_dG = __enzyme_fwddiff<serac::tensor<double, 3, 3>>((void*) serac::compute_stress, enzyme_const, &mat, 
        enzyme_dup, &state, &dstate, enzyme_dup, &H, &dH, enzyme_dup, K, dK, enzyme_dup, G, dG);

    // compute exact solution
    auto I = serac::DenseIdentity<3>();
    serac::tensor<double, 3, 3> B_minus_I = serac::dot(H, serac::transpose(H)) + serac::transpose(H) + H;
    double J = serac::det(I + H);
    auto exact = (B_minus_I - 2./3.*log(J)*I)/J;

    const double TOL = 1e-10*G;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_NEAR(dstress_dG[i][j], exact[i][j], TOL);
        }
    }
}

TEST(material, ParameterDerivativeWithState) {
    serac::J2 mat{.density=1.0};
    serac::J2::State state{};
    serac::tensor<double, 3, 3> H{{{0.19165388147925, 0.44595621086207, 0.03873204915048},
                                   {0.58923368584434, 0.0923605872371 , 0.25974694007571},
                                   {0.83097065578267, 0.48547287595839, 0.03308538443643}}};
    double E = 2.0;
    double nu = 0.25;
    double sigma_y = 0.01;
    double Hi = E/50.0;
    double Hk = 0.0;

    // serac::tensor<double, 3, 3> H{{{1.0, 0.0, 0.0},
    //                                {0.0, -0.25, 0.0},
    //                                {0.0, 0.0, -0.25}}};
    // H = 0.9*sigma_y/E*H;

    // tangent values
    serac::J2::State dstate{};
    serac::tensor<double, 3, 3> dH{};
    double dE = 0.0;
    double dnu = 0.0;
    double d_sigma_y = 0.0;
    double d_Hi = 1.0;
    double d_Hk = 0.0;

    std::cout << "dstate.beta = " << dstate.beta << std::endl;
    std::cout << "dstate.plastic_strain = " << dstate.plastic_strain << std::endl;
    std::cout << "dstate.eqps = " << dstate.accumulated_plastic_strain << std::endl;

    serac::tensor<double, 3, 3> stress;
    serac::tensor<double, 3, 3> dstress_dHi{};

    __enzyme_fwddiff<serac::tensor<double, 3, 3>>((void*) serac::J2_wrapper, enzyme_const, &mat, 
        enzyme_dup, &state, &dstate, enzyme_dup, &H, &dH, enzyme_dup, E, dE, enzyme_dup, nu, dnu, enzyme_dup, sigma_y, d_sigma_y, enzyme_dup, Hi, d_Hi, enzyme_dup, Hk, d_Hk, enzyme_dup, &stress, &dstress_dHi);

    std::cout << "stress = " << stress << std::endl;
    std::cout << "dstress_dHi = " << dstress_dHi << std::endl;
    std::cout << "eqps new = " << state.accumulated_plastic_strain << std::endl;
    std::cout << "deqps_dHi = " << dstate.accumulated_plastic_strain << std::endl;
    double G = 0.5*E/(1.0 + nu);
    double exact = -state.accumulated_plastic_strain/(3*G + Hk + Hi);
    std::cout << "exact = " << exact << std::endl;

    // TO DO: write exact solution or FD test
    // Force failure in meantime
    double error = std::abs(exact - dstate.accumulated_plastic_strain);
    EXPECT_LT(error, 1e-10);
}
