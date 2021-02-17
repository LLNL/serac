-- by construction, f(x, y, t) satisfies df_dt == d2f_dx2 + d2f_dy2
temp_func = function (v, t)
    return 1.0 + 6.0 * v.x * t - 2.0 * v.y * t + (v.x - v.y) * v.x * v.x
end

-- Comparison information
epsilon = 0.00005

exact_solution = {
    scalar_function = temp_func
}

-- Simulation time parameters
dt      = 0.5
t_final = 5.0

-- Simulation output format
output_type = "VisIt"

main_mesh = {
    type = "file",
    -- mesh file
    mesh = "../../../meshes/star.mesh",
    -- serial and parallel refinement levels
    ser_ref_levels = 2,
    par_ref_levels = 1,
}

-- Solver parameters
thermal_conduction = {
    stiffness_solver = {
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

    dynamics = {
        timestepper = "BackwardEuler",
        enforcement_method = "RateControl",
    },

    -- polynomial interpolation order
    order = 2,

    -- material parameters
    kappa = 1.0,
    -- rho = 0.5,
    -- cp = 0.5,

    -- initial conditions
    initial_temperature = {
        scalar_function = temp_func
    },

    -- boundary condition parameters
    boundary_conds = {
        ['temperature'] = {
            attrs = {1},
            scalar_function = temp_func
        },
    },
}
