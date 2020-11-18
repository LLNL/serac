-- Simulation time parameters
t_final = 1.0
dt      = 0.25

main_mesh = {
    -- mesh file
    mesh = "../meshes/beam-hex.mesh",
    -- serial and parallel refinement levels
    ser_ref_levels = 0,
    par_ref_levels = 0,
}

-- Solver parameters
nonlinear_solid = {
    stiffness_solver = {
        linear = {
            rel_tol     = 1.0e-6,
            abs_tol     = 1.0e-8,
            max_iter    = 5000,
            print_level = 0,
            solver_type = "gmres",
            prec_type   = "AMG",
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

    -- loading parameters
    traction = {
        x = 0.0,
        y = 1.0e-3,
        z = 0.0,
    },

    -- boundary condition parameters
    boundary_conds = {
        {
            name = "displacement",
            attrs = {1},
        },
        {
            name = "traction",
            attrs = {2},
        },
    },
}
