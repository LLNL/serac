-- _output_type_start
output_type = "ParaView"
-- _output_type_end

-- Mesh information
-- _mesh_start
main_mesh = {
    type = "generate",
    elements = {x = 10, y = 10}
}
-- _mesh_end

-- Solver parameters
thermal_conduction = {
  -- _solver_opts_start
    equation_solver = {
        linear = {
            type = "iterative",
            iterative_options = {
                rel_tol     = 1.0e-6,
                abs_tol     = 1.0e-12,
                max_iter    = 200,
                -- WARNING: Undocumented in MFEM, but ranges from -1 to 3,
                -- -1 is no output, 0 is warnings/errors only,
                -- 1 is per-iteration norm only, 2 is final iteration count only,
                -- 3 includes extra debug info
                print_level = 0,
                solver_type = "cg",
                prec_type   = "JacobiSmoother",
            },
        },

        nonlinear = {
            rel_tol     = 1.0e-4, -- w.r.t. residual
            abs_tol     = 1.0e-8,
            max_iter    = 500,
            -- WARNING: Undocumented in MFEM, but ranges from -1 to 3,
            -- -1 is no output, 0 is warnings/errors only,
            -- 1 is per-iteration norm only, 2 is final iteration count only,
            -- 3 includes extra debug info
            print_level = 1,
        },
    },

    order = 2,
    -- _solver_opts_end

    -- _conductivity_start
    kappa = 0.5,
    -- _conductivity_end

    -- _bc_start
    boundary_conds = {
        ['temperature_1'] = {
            attrs = {1},
            constant = 1.0
        },
        ['temperature_2'] = {
            attrs = {2, 3},
            scalar_function = function(v)
                return v.x * v.x + v.y - 1
            end
        }
    },
    -- _bc_end
}
