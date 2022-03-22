-- meemee Failing: error message:
--[[
ERROR (/usr/WS2/meemee/serac/repo/src/serac/numerics/equation_solver.cpp:147)]
AMGX was not enabled when MFEM was built
--]]                 
    
-- Comparison information
expected_temperature_l2norm = 2.02263
epsilon = 0.00001

-- Simulation time parameters
dt      = 1.0

main_mesh = {
    type = "ball",
    -- number of elements in the mesh
    approx_elements = 10000,
    -- serial and parallel refinement levels
    ser_ref_levels = 0,
    par_ref_levels = 0,
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
                prec_type   = "L1JacobiAMGX",
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

    -- boundary condition parameters
    boundary_conds = {
        ['temperature'] = {
            attrs = {1},
            constant = 1.0
        },
    },
}
