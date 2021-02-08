-- Comparison information
expected_t_l2norm = 1.99716
epsilon = 0.00001

-- Simulation time parameters
dt      = 1.0

main_mesh = {
    type = "file",
    -- mesh file
    mesh = "../../../meshes/star_with_2_bdr_attributes.mesh",
    -- serial and parallel refinement levels
    ser_ref_levels = 1,
    par_ref_levels = 1,
}

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

    -- add a nonlinear source
    nonlinear_reaction = {
        reaction_function = function (temp)
            return 0.1 * temp^2
        end,
        d_reaction_function = function (temp)
            return 0.2 * temp
        end
    },

    -- boundary condition parameters
    boundary_conds = {
        ['temperature'] = {
            attrs = {1},
            coef = function (v)
                return 1.0
            end
        },
    },
}
