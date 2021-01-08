-- Comparison information
expected_x_l2norm = 1.4225
expected_v_l2norm = 0.2252
epsilon = 0.0001


main_mesh = {
    -- mesh file
    mesh = "../../../meshes/beam-hex.mesh",
    -- serial and parallel refinement levels
    ser_ref_levels = 1,
    par_ref_levels = 0,
    width = 1.,
    len = 8.
}

-- material property helper functions
function linear_mu(E, nu)
    return 0.5 * E / (1. + nu);
end

function linear_bulk(E, nu)
    return E / ( 3. * (1. - 2.*nu));
end

-- Problem parameters
E = 17.e6;
nu = 0.3;
r = 1.; -- density
g = -32.3; -- gravity
i = 1./12 * main_mesh.width * main_mesh.width * main_mesh.width;
m = main_mesh.width * r;
omega = (1.875) * (1.875) * math.sqrt(E * i / (m * main_mesh.len * main_mesh.len * main_mesh.len * main_mesh.len));
period = 2. * math.pi / omega;

-- Simulation time parameters
t_final = 2. * period;
nsteps = 40;
dt      = t_final / nsteps;

print("t_final = " .. t_final);

-- Solver parameters
nonlinear_solid = {
    stiffness_solver = {
        linear = {
            type = "iterative",
            iterative_options = {
                rel_tol     = 1.0e-8,
                abs_tol     = 1.0e-12,
                max_iter    = 500,
                print_level = 0,
                solver_type = "gmres",
                prec_type   = "HypreAMG",
            },
        },

        nonlinear = {
            rel_tol     = 1.0e-8,
            abs_tol     = 1.0e-12,
            max_iter    = 500,
            print_level = 1,
        },
    },

    dynamics = {
        timestepper = "NewmarkBeta",
        enforcement_method = "RateControl",
    },

    -- polynomial interpolation order
    order = 1,

    -- neo-Hookean material parameters corresponding to lienar elasticity
    mu = linear_bulk(E, nu)/2.;
    K  = linear_bulk(E, nu)/2.;

--[[
    -- initial conditions
    initial_displacement = {
        vec_coef = function (x, y, z)
            return 0, 0
        end  
    },

    initial_velocity = {
        vec_coef = function (x, y, z)
            return 0, 1
        end 
    },

    --]]

    -- boundary condition parameters
    boundary_conds = {
        ['displacement'] = {
            -- boundary attribute 1 (index 0) is fixed (Dirichlet) in the x direction
            attrs = {2},
            vec_coef = function (x, y, z)
                return 0, 0, 0
            end 
        },
    },



}
