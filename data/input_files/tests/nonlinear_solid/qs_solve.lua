-- Comparison information
expected_x_l2norm = 2.2309025
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

-- Simulation output format
output_type = "VisIt"

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

    initial_displacement = {
        constant_vector = {
            x = 0.0,
            y = 0.0,
            z = 0.0
        }
    },

    initial_velocity = {
        constant_vector = {
            x = 0.0,
            y = 0.0,
            z = 0.0
        }
    }, 

    -- boundary condition parameters
    boundary_conds = {
        ['displacement'] = {
            -- boundary attribute 1 (index 0) is fixed (Dirichlet) in the x direction
            attrs = {1},
            constant_vector = {
                x = 0.0,
                y = 0.0,
                z = 0.0
            }
        
        },
        ['traction'] = {
            attrs = {2},
            constant_vector = {
                x = 0.0,
                y = 1.0e-3,
                z = 0.0
            }
        },
    },
}
