-- run the given set of options
function run_table(thermal_solver)
    -- get refined shared_ptr<ParMesh>
    mesh = MeshReader(thermal_solver.mesh.filename, thermal_solver.mesh.serial, thermal_solver.mesh.parallel)

    -- create solver
    solver = ThermalSolverBuilder(thermal_solver.order, 
                                  mesh, --TheramlSolver(2, pmesh)
                                  thermal_solver.timestepper, -- SetTimestepper
                                  thermal_solver.u0,
                                  thermal_solver.kappa,
                                  thermal_solver.solver.rel_tol,
                                  thermal_solver.solver.abs_tol,
                                  thermal_solver.solver.print_level,
                                  thermal_solver.solver.max_iter
                                )   
    rawset(_G, "solver", solver)

    if (thermal_solver.solver.steps) then
       for ti = 0, thermal_solver.solver.steps, 1
       do
              ThermalSolverStep(solver, thermal_solver.solver.dt)
       end
    end

end
