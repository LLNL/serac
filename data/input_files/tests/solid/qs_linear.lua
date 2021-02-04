-- Comparison information
expected_x_l2norm = 2.4276439
epsilon = 0.0001

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
solid = {
    stiffness_solver = {
        -- Use a direct solver to check for machine precision convergence in one Newton step
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
    mu = 0.25,
    K  = 10.0,

    -- Turn the geometric nonlinearities off
    geometric_nonlin = false,

    -- Turn the material nonlinearities off
    -- TODO: this should be replaced with a proper material definition
    material_nonlin = false,

    -- boundary condition parameters
    boundary_conds = {
        ['displacement'] = {
            -- boundary attribute 1 is fixed (Dirichlet)
            attrs = {1},
            vec_coef = function (v)
                return Vector.new(0, 0, 0)
            end
        },
        ['traction'] = {
            -- boundary attribute 1 (index 0) is fixed (Dirichlet) in the x direction
            attrs = {2},
            vec_coef = function (v)
                return Vector.new(0, -1.0e-3, 0)
            end
        },
    },
}
