#include <gtest/gtest.h>

#include "serac/numerics/functional/quadrature_data.hpp"
#include "serac/physics/materials/solid_material.hpp"

using namespace serac;

using material_model = solid_mechanics::J2;
using mat3 = tensor<double, 3, 3>;

template < typename material >
std::function< QData(const std::vector<mat3> &, const QData &) > func(material model) {
    return [model](const std::vector< mat3 > & du_dX, const QData & internal_variables) {
        using state_type = typename material::State;

        QData next_internal_variables = internal_variables;
        span2D<state_type> iv = span<state_type>(next_internal_variables);

        uint32_t num_elems = iv.dimensions[0];
        uint32_t qpts_per_elem = iv.dimensions[1];
        uint32_t i = 0;
        for (uint32_t e = 0; e < num_elems; e++) {
            for (uint32_t q = 0; q < qpts_per_elem; q++) {
                model(iv(e, q), du_dX[i]);
                i++;
            }
        }
        return next_internal_variables;
    };
}

int main() {

    using state_type = material_model::State;

    uint32_t nelems = 5;
    uint32_t qpts_per_elem = 2;
    uint32_t num_qpts = nelems * qpts_per_elem;

    std::vector< state_type > initial_state(num_qpts, state_type{});

    mat3 H = {{
        {0.0, 0.1, 0.0},
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0}
    }};
    std::vector< mat3 > du_dX(num_qpts, H);
    serac::QData internal_variables(initial_state, {nelems, qpts_per_elem});

    material_model mat{
        100.0, ///< Young's modulus
        0.25,  ///< Poisson's ratio
        0.0,   ///< isotropic hardening constant
        0.0,   ///< kinematic hardening constant
        40.0,  ///< yield stress
        1.0    ///< mass density
    };

    auto apply_deformation = func(mat);

    for (int i = 1; i < 10; i++) {
        du_dX = std::vector< mat3 >(num_qpts, H * i);
        internal_variables = apply_deformation(du_dX, internal_variables);
        span2D<state_type> iv = span<state_type>(internal_variables);
        std::cout << i << " " << iv(0,0).accumulated_plastic_strain << std::endl;
    }

}