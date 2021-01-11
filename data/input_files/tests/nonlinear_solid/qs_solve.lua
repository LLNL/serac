-- Comparison information
expected_x_l2norm = 2.2309025
epsilon = 0.001

-- Simulation time parameters
dt      = 1.0

main_mesh = {
    -- mesh file
    mesh = "../../../meshes/beam-hex.mesh",
    -- serial and parallel refinement levels
    ser_ref_levels = 1,
    par_ref_levels = 0,
}

-- Solver parameters
nonlinear_solid = {
    stiffness_solver = {
        linear = {
            type = "iterative",
            iterative_options = {
                rel_tol     = 1.0e-6,
                abs_tol     = 1.0e-8,
                max_iter    = 5000,
                print_level = 0,
                solver_type = "minres",
                prec_type   = "L1JacobiSmoother",
            },
        },

        nonlinear = {
            rel_tol     = 1.0e-3,
            abs_tol     = 1.0e-6,
            max_iter    = 5000,
            print_level = 1,
        },
    },

    -- polynomial interpolation order
    order = 1,

    -- neo-Hookean material parameters
    mu = 0.25,
    K  = 10.0,

    -- boundary condition parameters
    boundary_conds = {
        ['displacement'] = {
            -- boundary attribute 1 (index 0) is fixed (Dirichlet) in the x direction
            attrs = {1},
            vec_coef = function (x, y, z)
                return 0, 0, 0
            end
        },
        ['traction'] = {
            attrs = {2},
            vec_coef = function (x, y, z)
                return 0, 1.0e-3, 0
            end
        },
    },
}
