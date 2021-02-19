-- Experimental data
experimental_flux_coef = {
  scalar_function = function (position)
    return position.x * 0.1
  end
}

bottom_temp  = function (position)
  return 200 + position.x * 0.5
end

-- Regularization parameter
epsilon = 1.0e-4 

-- Boundary_markers
measured_boundary = 2
unknown_boundary = 1

-- Initialize the design field with a guess
initial_top_flux_guess = {
  constant = 1.0
}

main_mesh = {
    type = "file",
    -- mesh file
    mesh = "../meshes/square.mesh",
    -- serial and parallel refinement levels
    ser_ref_levels = 1,
    par_ref_levels = 1,
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
                prec_type   = "JacobiSmoother",
            },
        },
    },

    -- polynomial interpolation order
    order = 1,

    -- material parameters
    kappa = 1.0,

    -- boundary condition parameters
    -- NOTE: the driver code knows how to add the flux term
    boundary_conds = {
        ['temperature'] = {
            attrs = {measured_boundary},
            scalar_function = bottom_temp
        },
    },
}
