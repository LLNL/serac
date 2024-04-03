-- Comparison information
expected_displacement_l2norm = 2.2309025
epsilon = 0.001

-- Simulation time parameters
dt      = 1.0

main_mesh = {
    type = "file",
    -- mesh file
    mesh = "../../../meshes/beam-hex.mesh",
    -- serial and parallel refinement levels
    ser_ref_levels = 1,
    par_ref_levels = 0,
}

-- Solver parameters
solid = {
    equation_solver = {
        linear = {
            type = "direct",
            direct_options = {
                print_level = 0,
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
    materials = { { model = "NeoHookean", mu = 0.25, K = 10.0, density = 1.0 }, },

    -- boundary condition parameters
    boundary_conds = {
        ['displacement'] = {
            -- boundary attribute 1 (index 0) is fixed (Dirichlet) in the x direction
            attrs = {1},
            vector_constant = {
                x = 0.0,
                y = 0.0,
                z = 0.0
            }
        },
        ['traction'] = {
            attrs = {2},
            vector_constant = {
                x = 0.0,
                y = 1.0e-3,
                z = 0.0
            }
        },
    },
}
