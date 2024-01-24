#include "enzyme.hpp"

#include "tensor.hpp"
//#include "materials.hpp"

using serac::tensor;
using serac::DenseIdentity;

using vec3 = serac::tensor<double,3>;
using mat3 = serac::tensor<double,3,3>;

struct State { mat3 R; };

static constexpr int num_parameters = 3;
static constexpr int num_state_variables = 9;

struct PathologicalMaterial {

  mat3 operator()(/*State& state, */const mat3 & du_dX, const vec3 & parameters) {

    const auto [kx, ky, kz] = parameters;

    mat3 K = {{{  0, -kz, +ky},
               {+kz,   0, -kx},
               {-ky, +kx,   0}}};

    // this is a stupid way to generate a rotation matrix
    // from an axis-angle representation, but it does generate 
    // quite a bit of calculation in a for loop, to stress test
    // enzyme's reverse-mode differentiation
    mat3 R{};
    mat3 KN = DenseIdentity<3>();
    double ifactorial = 1;
    for (int i = 0; i < 30; i++) {
        R += KN / ifactorial;
        KN = dot(K, KN);
        ifactorial = (i == 0) ? 1 : ifactorial * (i+1);
    }

    //state.R = dot(state.R, R);

    return dot(R, du_dX);

  }

};

template < typename T, typename ... arg_types >
auto wrapper1(T obj, arg_types ... args) {
    return obj(args ... );
}

template < typename T, typename ... arg_types >
auto wrapper2(T obj, arg_types & ... args) {
    return obj(args ... );
}

template < typename T, typename ... arg_types >
auto wrapper3(T obj, arg_types && ... args) {
    return obj(args ... );
}

mat3 baz(mat3 y, mat3 z) { return dot(y, z) + z; }

struct MyObject1{
    double operator()(double y, double z) { return x + y * z; }
    double x;
};

struct MyObject2{
    double operator()(vec3 y, vec3 z) { return x + dot(y, z); }
    double x;
};

struct MyObject3{
    mat3 operator()(mat3 y, mat3 z) { return x * dot(y, z); }
    double x;
};

struct MyObject4{
    vec3 operator()(State S, mat3 & y, vec3 z) { return x * dot(y, z); }
    double x;
};

int main() {

    using mat_t = PathologicalMaterial;
    using state_t = State;

    mat_t mat{};
    state_t state{DenseIdentity<3>()};

    mat3 du_dX{{{1,2,3},
                {4,5,6},
                {7,8,9}}};

    vec3 k{0.0, 1.0, 2.0};

    vec3 dk{0.0, 1.0, 0.0};

    mat(du_dX, k);

#define WHICH 0
#if WHICH==0
    mat3 output = __enzyme_fwddiff<mat3>((void*)wrapper2<mat_t, mat3, vec3>, 
        enzyme_const, mat,
        //enzyme_const, &state,
        enzyme_const, &du_dX, 
        enzyme_dup, &k, &dk);
#endif

#if WHICH==1
    MyObject1 obj{1.0};
    double y = 1.0;
    double z = 1.0;
    double dz = 1.0;
    double dfdy = __enzyme_fwddiff<double>((void*)wrapper1<MyObject1, double, double>, enzyme_const, obj, 
                                                                                       enzyme_const, y,
                                                                                       enzyme_dup, z, dz);

    dfdy = __enzyme_fwddiff<double>((void*)wrapper2<MyObject1, double, double>, enzyme_const, obj, 
                                                                                enzyme_const, &y,
                                                                                enzyme_dup, &z, &dz);

    dfdy = __enzyme_fwddiff<double>((void*)wrapper3<MyObject1, double, double>, enzyme_const, obj, 
                                                                                enzyme_const, &y,
                                                                                enzyme_dup, &z, &dz);

    std::cout << dfdy << std::endl;

#endif

#if WHICH==2
    MyObject2 obj{1.0};
    vec3 y = {1.0, 2.0, 3.0};
    vec3 z = {2.0, 3.0, 4.0};
    vec3 dz = {0.0, 1.0, 1.0};
    double dfdy = __enzyme_fwddiff<double>((void*)wrapper1<MyObject2, vec3, vec3>, enzyme_const, obj, 
                                                                                   enzyme_const, y,
                                                                                   enzyme_dup, z, dz);

    dfdy = __enzyme_fwddiff<double>((void*)wrapper2<MyObject2, vec3, vec3>, enzyme_const, obj, 
                                                                            enzyme_const, &y,
                                                                            enzyme_dup, &z, &dz);

    dfdy = __enzyme_fwddiff<double>((void*)wrapper3<MyObject2, vec3, vec3>, enzyme_const, obj, 
                                                                            enzyme_const, &y,
                                                                            enzyme_dup, &z, &dz);
    std::cout << dfdy << std::endl;
#endif

#if WHICH==3
    MyObject3 obj{1.0};
    mat3 y{};
    mat3 z{};
    mat3 dz{};
    mat3 dfdy = __enzyme_fwddiff<mat3>((void*)wrapper1<MyObject3, mat3, mat3>, enzyme_const, obj, 
                                                                               enzyme_const, y,
                                                                               enzyme_dup, z, dz);

    dfdy = __enzyme_fwddiff<mat3>((void*)wrapper2<MyObject3, mat3, mat3>, enzyme_const, obj, 
                                                                          enzyme_const, &y,
                                                                          enzyme_dup, &z, &dz);

    dfdy = __enzyme_fwddiff<mat3>((void*)wrapper3<MyObject3, mat3, mat3>, enzyme_const, obj, 
                                                                          enzyme_const, &y,
                                                                          enzyme_dup, &z, &dz);
    std::cout << dfdy << std::endl;
#endif


#if WHICH==4
    MyObject4 obj{1.0};
    mat3 y{};
    vec3 z{};
    vec3 dz{};
    vec3 dfdy = __enzyme_fwddiff<vec3>((void*)wrapper1<MyObject4, mat3, vec3>, enzyme_const, obj, 
                                                                               enzyme_const, y,
                                                                               enzyme_dup, z, dz);

    dfdy = __enzyme_fwddiff<vec3>((void*)wrapper2<MyObject4, mat3, vec3>, enzyme_const, obj, 
                                                                          enzyme_const, &y,
                                                                          enzyme_dup, &z, &dz);

    dfdy = __enzyme_fwddiff<vec3>((void*)wrapper3<MyObject4, mat3, vec3>, enzyme_const, obj, 
                                                                          enzyme_const, &y,
                                                                          enzyme_dup, &z, &dz);
    std::cout << dfdy << std::endl;
#endif




}