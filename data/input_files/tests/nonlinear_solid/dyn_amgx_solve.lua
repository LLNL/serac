-- Comparison information
expected_x_l2norm = 1.4225
expected_v_l2norm = 0.2252
epsilon = 0.0001

-- Simulation time parameters
dt      = 1.0
t_final = 6.0

main_mesh = {
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
                rel_tol     = 1.0e-4,
                abs_tol     = 1.0e-8,
                max_iter    = 500,
                print_level = 0,
                solver_type = "gmres",
                prec_type   = "AMGX",
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
        timestepper = "AverageAcceleration",
        enforcement_method = "RateControl",
    },

    -- polynomial interpolation order
    order = 1,

    -- neo-Hookean material parameters
    mu = 0.25,
    K  = 5.0,

    viscosity = 0.0,

    -- initial conditions
    initial_displacement = {
        vec_coef = function (x, y, z)
            return 0, 0, 0
        end  
    },

    initial_velocity = {
        vec_coef = function (x, y, z)
            s = 0.1 / 64
            first = -s * x * x
            last = s * x * x * (8.0 - x)
            -- FIXME: How can we detect the dimension?
            return first, 0, last
        end 
    },

    -- boundary condition parameters
    boundary_conds = {
        ['displacement'] = {
            -- boundary attribute 1 (index 0) is fixed (Dirichlet) in the x direction
            attrs = {1},
            vec_coef = function (x, y, z)
                return 0, 0, 0
            end 
        },
    },
}
