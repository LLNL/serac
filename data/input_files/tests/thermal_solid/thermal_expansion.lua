-- Simulation time parameters
t_final = 1.0
dt      = 1.0

expected_u_l2norm = 0.15090898
expected_v_l2norm = 0.0
expected_t_l2norm = 2.344805
epsilon = 0.0001

output_type = "SidreVisIt"

main_mesh = {
    type = "file",
    -- mesh file
    mesh = "../../../meshes/onehex.mesh",
    -- serial and parallel refinement levels
    ser_ref_levels = 1,
    par_ref_levels = 0,
}

thermal_solid = {
    -- Solver parameters
    solid = {
        equation_solver = {
            linear = {
                type = "iterative",
                iterative_options = {
                    rel_tol     = 1.0e-6,
                    abs_tol     = 1.0e-8,
                    max_iter    = 5000,
                    print_level = 0,
                    solver_type = "gmres",
                    prec_type   = "HypreAMG",
                },
            },

            nonlinear = {
                rel_tol     = 1.0e-2,
                abs_tol     = 1.0e-4,
                max_iter    = 500,
                print_level = 0,
            },
        },

        -- polynomial interpolation order
        order = 1,

        -- neo-Hookean material parameters
        mu = 0.25,
        K  = 5.0,

        initial_displacement = {
            vector_constant = {
                x = 0.0,
                y = 0.0,
                z = 0.0
            }
        },

        initial_velocity = {
            vector_constant = {
                x = 0.0,
                y = 0.0,
                z = 0.0
            }
        },

        -- boundary condition parameters
        boundary_conds = {
            ['displacement'] = {
                attrs = {1},
                component = 0,
                constant = 0.0
            },
            ['displacement'] = {
                attrs = {2},
                component = 1,
                constant = 0.0,
            },
            ['displacement'] = {
                attrs = {3},
                component = 2,
                constant = 0.0,
            },
        },
    },

    -- Solver parameters
    thermal_conduction = {
        equation_solver = {
            linear = {
                type = "iterative",
                iterative_options = {
                    rel_tol     = 1.0e-6,
                    abs_tol     = 1.0e-12,
                    max_iter    = 200,
                    print_level = 0,
                    solver_type = "cg",
                    prec_type   = "JacobiSmoother",
                },
            },

            nonlinear = {
                rel_tol     = 1.0e-4,
                abs_tol     = 1.0e-8,
                max_iter    = 500,
                print_level = 1,
            },
        },

        -- polynomial interpolation order
        order = 2,

        -- material parameters
        kappa = 0.5,
        rho = 0.5,
        cp = 0.5,

        -- initial conditions
        initial_temperature = {
            constant = 2.0
        },

        -- boundary condition parameters
        boundary_conds = {
            ['temperature'] = {
                attrs = {1,2,3,4,5,6},
                constant = 2.0
            },
        },
    },

    coef_thermal_expansion = {
        constant = 0.1
    },

    reference_temperature = {
        constant = 1.0
    }

}
