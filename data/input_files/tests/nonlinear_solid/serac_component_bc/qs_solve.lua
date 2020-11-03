-- Comparison information
expected_l2norm = 0.08363646
epsilon = 0.0001

-- Simulation time parameters
dt      = 1.0

main_mesh = {
    -- mesh file
    mesh = "../../../../meshes/square.mesh",
    -- serial and parallel refinement levels
    ser_ref_levels = 2,
    par_ref_levels = 0,
}

-- Solver parameters
nonlinear_solid = {
    solver = {
        linear = {
            rel_tol     = 1.0e-8,
            abs_tol     = 1.0e-12,
            max_iter    = 5000,
            print_level = 0,
            solver_type = "minres",
            prec_type   = "L1JacobiSmoother",
        },

        nonlinear = {
            rel_tol     = 1.0e-6,
            abs_tol     = 1.0e-8,
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
        {
            name = "displacement",
            -- boundary attribute 1 (index 0) is fixed (Dirichlet) in the x direction
            attrs = {1},
        },
    },
}
