-- Simulation output format
output_type = "VisIt"

-- Mesh information
main_mesh = {
  type = "generate",
  -- 10-by-10 quad mesh
  elements = {x = 10, y = 10}
}

-- Solver parameters
thermal_conduction = {
  stiffness_solver = {
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

  -- polynomial interpolation order
  order = 2,

  -- material parameters
  kappa = 0.5,

  -- boundary condition parameters
  boundary_conds = {
      ['temperature'] = {
          attrs = {1},
          constant = 1.0
      },
  },
}
