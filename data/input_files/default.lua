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

-- Materials
material_j2 = { model = "J2", E = 100.0, nu = 0.25, hardening = 100.0 } -- TODO chapman39 create input struct for Hardening?
material_linear_isotropic_conductor = {
    model = "LinearIsotropicConductor",
    kappa = 0.7,
    cp = 1.5,
    density = 0.5,
}

-- Solver parameters
solid = {
    equation_solver = {
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
    order = 2,

    -- material parameters
    materials = {
        { model = "NeoHookean", mu = 0.26, K = 5.5, density = 2 },
        material_j2
    },

    -- initial conditions
    -- initialize x_cur, boundary condition, deformation, and
    -- incremental nodal displacment grid functions by projecting the
    -- VectorFunctionCoefficient function onto them

    initial_displacement = {
        vector_constant = {
            x = 0.0,
            y = 0.0,
            z = 0.0
        }
    },

    initial_velocity = {
        vector_constant = {
            x = 0.0,
            y = 0.0,
            z = 0.0
        }
    },

    -- boundary condition parameters
    boundary_conds = {
        ['displacement'] = {
            attrs = {1},
            vector_constant = {
                x = 0.0,
                y = 0.0,
                z = 0.0
            }
        },
    },
}

temp_func = function (v)
    if v:norm() < 3.0 then return 2.0 else return 1.0 end
end

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
    materials = {
        material_linear_isotropic_conductor
    },

    -- initial conditions
    initial_temperature = {
        scalar_function = temp_func
    },

    -- boundary condition parameters
    boundary_conds = {
        ['temperature'] = {
            attrs = {1},
            scalar_function = temp_func
        },
    },
}
