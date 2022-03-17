-- meemee Failing: error message:
--[[
ERROR (/usr/WS2/meemee/serac/repo/src/serac/infrastructure/input.cpp:332)]
Coefficient definition does not contain known type.

Frame 1: axom::slic::logErrorMessage(std::string const&, std::string const&, int)
Frame 2: FromInlet<serac::input::CoefficientInputOptions>::operator()(axom::inlet::Container const&)
Frame 4: FromInlet<serac::input::BoundaryConditionInputOptions>::operator()(axom::inlet::Container const&)
Frame 6: std::unordered_map<std::string, serac::input::BoundaryConditionInputOptions, std::hash<std::string>, std::equal_to<std::string>, std::allocator<std::pair<std::string const, serac::input::BoundaryConditionInputOptions> > > axom::inlet::Container::getCollection<std::string, serac::input::BoundaryConditionInputOptions>() const
Frame 9: FromInlet<serac::ThermalConduction::InputOptions>::operator()(axom::inlet::Container const&)
Frame 14: ../../build-quartz-toss_3_x86_64_ib-clang@10.0.0-debug/bin/serac() [0x769189]
--]]                 
-- also note: this is the only test that times out.

-- Comparison information
exact_function = function (v)
  return v.x^2 * v.y
end

minus_laplacian_temp = function (v)
  return -2.0 * v.y
end

reaction = function (temp)
  return temp^2 + 5.0
end

d_reaction = function (temp)
  return 2.0 * temp
end

scale_function = function (v)
    return math.sin(v.x) + math.cos(v.y)
end

exact_solution = {
  coef = exact_function
}

epsilon = 0.00001

-- Simulation time parameters
dt      = 1.0

main_mesh = {
    type = "file",
    -- mesh file
    mesh = "../../../meshes/star.mesh",
    -- serial and parallel refinement levels
    ser_ref_levels = 1,
    par_ref_levels = 1,
}

-- Solver parameters
thermal_conduction = {
    equation_solver = {
        linear = {
            type = "direct",
            direct_options = {
                print_level = 0,
            },
        },

        nonlinear = {
            rel_tol     = 1.0e-8,
            abs_tol     = 1.0e-12,
            max_iter    = 500,
            print_level = 1,
        },
    },

    -- polynomial interpolation order
    order = 3,

    -- material parameters
    kappa = 1.0,

    -- add a nonlinear reaction
    nonlinear_reaction = {
        reaction_function = reaction,
        d_reaction_function = d_reaction,
        scale = {
            coef = scale_function
        }
    },

    source = {
        coef = function (v)
            return scale_function(v) * reaction(exact_function(v)) + minus_laplacian_temp(v)
        end
    },

    -- boundary condition parameters
    boundary_conds = {
        ['temperature'] = {
            attrs = {1},
            coef = exact_function
        },
    },
}
