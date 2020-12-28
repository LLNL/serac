-- Comparison information
expected_t_l2norm = 2.1806652643
epsilon = 0.00001

-- Simulation time parameters
dt      = 1.0
t_final = 5.0

main_mesh = {
    -- mesh file
    mesh = "../../../meshes/star.mesh",
    -- serial and parallel refinement levels
    ser_ref_levels = 1,
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
    kappa = 0.5,
    rho = 0.5,
    cp = 0.5,

    -- initial conditions
    initial_temperature = {
        coef = function (x, y, z)
            local norm = math.sqrt(x ^ 2 + y ^ 2 + z ^ 2)
            if norm < 0.5 then return 2.0 else return 1.0 end
        end
    },

    -- boundary condition parameters
    boundary_conds = {
        ['temperature'] = {
            attrs = {1},
            coef = function (x, y, z)
                local norm = math.sqrt(x ^ 2 + y ^ 2 + z ^ 2)
                if norm < 0.5 then return 2.0 else return 1.0 end
            end
        },
    },
}
