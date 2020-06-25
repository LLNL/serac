-- a simple test to accompany serac_lua_test that matches dyn_imp_solve
--serac-thermal-solver
thermal_solver={}
thermal_solver.mesh = {
   filename = "/data/star.mesh",
   serial = 1,
   parallel = 1
}
thermal_solver.order = 2
thermal_solver.timestepper = "backwardeuler"

-- define initial conditions
thermal_solver.u0 = { type = "function", func = "InitialTemperature" }

-- define conducitivity
thermal_solver.kappa = { type = "constant", constant = 0.5}

-- linear solver parameters
thermal_solver.solver = {
    rel_tol = 1.e-6,
    abs_tol = 1.e-12,
    print_level = 0,
    max_iter = 100,
    dt = 1.0,
    t_final = 5.0
}


