-- Comparison information
expected_x_l2norm = 1.4225
expected_v_l2norm = 0.2252
epsilon = 0.0001

-- Simulation time parameters
dt      = 1.0
t_final = 6.0

main_mesh = {
    -- mesh file
    mesh = "../../../../meshes/beam-hex.mesh",
    -- serial and parallel refinement levels
    ser_ref_levels = 1,
    par_ref_levels = 0,
}

-- Solver parameters
nonlinear_solid = {
    stiffness_solver = {
        linear = {
            rel_tol     = 1.0e-4,
            abs_tol     = 1.0e-8,
            max_iter    = 500,
            print_level = 0,
            solver_type = "gmres",
            prec_type   = "AMGX",
        },

        nonlinear = {
            rel_tol     = 1.0e-4,
            abs_tol     = 1.0e-8,
            max_iter    = 500,
            print_level = 1,
        },
    },

    mass_solver = {
        timestepper = "AverageAcceleration",
        enforcement_method = "RateControl",
    },

    -- polynomial interpolation order
    order = 1,

    -- neo-Hookean material parameters
    mu = 0.25,
    K  = 5.0,

    -- boundary condition parameters
    boundary_conds = {
        {
            name = "displacement",
            -- boundary attribute 1 (index 0) is fixed (Dirichlet) in the x direction
            attrs = {1},
        },
    },
}
