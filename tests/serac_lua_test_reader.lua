-- run the given set of options

function BuildCoefficient(coefficient)
   if coefficient.type == "constant" then
      return CoefficientFactory("constant", coefficient.constant)
   elseif coefficient.type == "function" then
      return CoefficientFactory("function", coefficient.func)
   else
	error("not supported ")
   end
end

function run_table(thermal_solver)
    -- get refined shared_ptr<ParMesh>
    mesh = MeshReader(thermal_solver.mesh.filename, thermal_solver.mesh.serial, thermal_solver.mesh.parallel)

    -- create function
    u0 = BuildCoefficient(thermal_solver.u0)
    kappa = BuildCoefficient(thermal_solver.kappa)    

    -- create solver
    solver = ThermalSolverBuilder(thermal_solver.order, 
                                  mesh, --ThermalSolver(2, pmesh)
                                  thermal_solver.timestepper, -- SetTimestepper
                                  u0,
                                  kappa,
                                  thermal_solver.solver.rel_tol,
                                  thermal_solver.solver.abs_tol,
                                  thermal_solver.solver.print_level,
                                  thermal_solver.solver.max_iter
                                )   
    rawset(_G, "solver", solver)

    if (thermal_solver.solver.steps) then
       t = 0
       for ti = 0, thermal_solver.solver.steps, 1
       do
           dt = ThermalSolverStep(solver, thermal_solver.solver.dt)
	   t = t + dt
       end
    elseif (thermal_solver.solver.t_final) then
       t = 0
       dt = thermal_solver.solver.dt
       t_final = thermal_solver.solver.t_final
       while (t <= t_final - 1.e-8 * dt)
       do
           dt = math.min(dt, t_final - t)
           dt = ThermalSolverStep(solver, dt)
	   t = t + dt
	   print ("current time" .. tostring(t))
       end       
    end

end
