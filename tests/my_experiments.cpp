#include "gtest/gtest.h"

//#include "tuple.hpp"
#include "tensor.hpp"
#include "tuple_tensor_dual_functions.hpp"

int enzyme_dup;
int enzyme_out;
int enzyme_const;

namespace serac {

struct Empty {
};

struct NeoHookean {
  using State = Empty;  ///< this material has no internal variables

  template <typename T>
  SERAC_HOST_DEVICE auto operator()(State& /* state */, const tensor<T, 3, 3>& du_dX, double K, double G) const
  {
    using std::log;
    constexpr auto I         = DenseIdentity<3>();
    auto           lambda    = K - (2.0 / 3.0) * G;
    auto           B_minus_I = du_dX * transpose(du_dX) + transpose(du_dX) + du_dX;
    auto           J         = det(I + du_dX);
    return (lambda * log(J) * I + G * B_minus_I) / J;
  }

  double density;  ///< mass density
};

} // namespace serac

TEST(material, dummy) {
    serac::tensor<double, 3, 3> H{{{0.19165388147925, 0.44595621086207, 0.03873204915048},
                                   {0.58923368584434, 0.0923605872371 , 0.25974694007571},
                                   {0.83097065578267, 0.48547287595839, 0.03308538443643}}};
    serac::NeoHookean mat{.density=1.0};
    serac::NeoHookean::State state;
    auto stress = mat(state, H, 1.0, 1.0);
    EXPECT_NE(stress[0][0], 0.0);
}