extern int enzyme_dup;
extern int enzyme_dupnoneed;
extern int enzyme_out;
extern int enzyme_const;

template < typename return_type, typename ... T >
return_type __enzyme_fwddiff(void*, T ... );

template < typename return_type, typename ... T >
return_type __enzyme_autodiff(void*, T ... );

struct vec3{ double x, y, z; };
struct mat3{ double data[3][3]; };

vec3 dot(mat3 A, vec3 v) {
    return vec3{
        A.data[0][0] * v.x + A.data[0][1] * v.y + A.data[0][2] * v.z,
        A.data[1][0] * v.x + A.data[1][1] * v.y + A.data[1][2] * v.z,
        A.data[2][0] * v.x + A.data[2][1] * v.y + A.data[2][2] * v.z
    };
}

mat3 dot(mat3 A, mat3 B) {
    mat3 C{};
    for (int i = 0; i <3; i++) {
        for (int j = 0; j <3; j++) {
            for (int k = 0; k <3; k++) {
                C.data[i][j] += A.data[i][k] * B.data[k][j];
            }
        }
    }
    return C;
}

mat3 operator/(mat3 A, double denom) {
    mat3 C{};
    for (int i = 0; i <3; i++) {
        for (int j = 0; j <3; j++) {
            C.data[i][j] = A.data[i][j] / denom;
        }
    }
    return C;
}

void operator+=(mat3 A, mat3 B) {
    for (int i = 0; i <3; i++) {
        for (int j = 0; j <3; j++) {
            A.data[i][j] += B.data[i][j];
        }
    }
}

struct Material {

  mat3 operator()(const mat3 & du_dX, const vec3 & parameters) {

    const auto [kx, ky, kz] = parameters;

    mat3 K = {{{  0, -kz, +ky},
              {+kz,   0, -kx},
              {-ky, +kx,   0}}};

    mat3 R{};
    mat3 KN = {{{1, 0, 0}, {0,1,0}, {0,0,1}}};
    double ifactorial = 1;
    for (int i = 0; i < 30; i++) {
        R += KN / ifactorial;
        KN = dot(K, KN);
        ifactorial = (i == 0) ? 1 : ifactorial * (i+1);
    }

    return dot(R, du_dX);
  }

};

template < typename T, typename ... arg_types >
auto wrapper(T obj, arg_types & ... args) {
    return obj(args ... );
}

int main() {

    Material mat{};

    mat3 du_dX{{{1,2,3},
                {4,5,6},
                {7,8,9}}};

    vec3 k{0.0, 1.0, 2.0};

    vec3 dk{0.0, 1.0, 0.0};

    mat(du_dX, k);

    mat3 output = __enzyme_fwddiff<mat3>((void*)wrapper<Material, mat3, vec3>, 
        enzyme_const, mat,
        enzyme_const, &du_dX, 
        enzyme_dup, &k, &dk);

}