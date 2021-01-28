-- Simulation time parameters
t_final = 1.0
dt      = 0.25

main_mesh = {
    type = "file",
    -- mesh file
    mesh = "../meshes/beam-hex.mesh",
    -- serial and parallel refinement levels
    ser_ref_levels = 0,
    par_ref_levels = 0,
}

-- Solver parameters
solid = {
    stiffness_solver = {
        linear = {
            type = "iterative",
            iterative_options = {
                rel_tol     = 1.0e-6,
                abs_tol     = 1.0e-8,
                max_iter    = 5000,
                print_level = 0,
                solver_type = "gmres",
                prec_type   = "HypreAMG",
            },
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

    -- initial conditions
    -- initialize x_cur, boundary condition, deformation, and
    -- incremental nodal displacment grid functions by projecting the
    -- VectorFunctionCoefficient function onto them

    initial_displacement = {
        vec_coef = function (x, y, z)
            return 0, 0, 0
        end  
    },

    initial_velocity = {
        vec_coef = function (x, y, z)
            return 0, 0, 0
        end 
    },

    -- boundary condition parameters
    boundary_conds = {
        ['displacement'] = {
            attrs = {1},
            vec_coef = function (x, y, z)
                return 0, 0, 0
            end
        },
        ['traction'] = {
            attrs = {2},
            vec_coef = function (x, y, z)
                return 0, 1.0e-3, 0
            end
            -- FIXME: Move time-scaling logic to Lua once arbitrary function signatures are allowed
            -- vec_coef = function (x, y, z, t)
            --     return 0 * t, 1.0e-3 * t, 0 * t
            -- end
        },
    },
}
