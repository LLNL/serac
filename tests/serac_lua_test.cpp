#include <gtest/gtest.h>
#include <sys/stat.h>
#include <stdio.h>
#include <string.h>
#include <lua.hpp>
#include <lauxlib.h>
#include <lualib.h>
#include <math.h>
#include <memory>
#include <iostream>
#include <mfem.hpp>
#include "serac_config.hpp"
#include "solvers/thermal_solver.hpp"

namespace luautil {
  /// Simple method of registering some Lua types (metatables)
  void RegisterLuaType(lua_State *state, std::string type_name) {
    lua_newtable(state);
    luaL_newmetatable(state, type_name.c_str());
    
    // create an empty meta table to represent ParMesh
    lua_pushliteral(state, "__index");
    lua_pushvalue(state, -2);
    lua_rawset(state, -3);
    lua_setglobal(state, type_name.c_str());
  }

  /// Convenience function Register a vector of type names
  void RegisterLuaTypes(lua_State *state, std::vector<std::string> type_names)
  {
    for (auto type_name : type_names)
      RegisterLuaType(state, type_name);
  }
  
  /**
     \brief  This will create a pointer to a shared_ptr that we can then pass around

     \param[in] state The lua state
     \param[in] shared The created shared_ptr to give to Lua
     \param[in] type_name The Lua type name
  */
  template <class T>
  int CreateLuaObject(lua_State *state, std::shared_ptr<T> shared, std::string type_name) {
    // now the magic happens
    void * userData = lua_newuserdata(state, sizeof(std::shared_ptr<T>));

    // Handle allocation failure
    if (!userData) {    
      std::cout << "failed to allocate userData" << std::endl;
      return 0;
    }

    new(userData) std::shared_ptr<T>(shared);

    // set appropriate registered metatable
    luaL_getmetatable(state, type_name.c_str());
    lua_setmetatable(state, -2);
    return 1;
  }

  /**
     \brief This method tires to get a specific pointer at a given index

     \param[in] state The lua state
     \param[in] index The parameter index from a lua call
     \param[in[ type_name The type_name to verify against
   */
  template <class T>
  std::shared_ptr<T> GetPointer(lua_State *L, int index, std::string type_name)
  {
    void * ptr = luaL_checkudata(L, index, type_name.c_str());
    if  (ptr)
      {
      auto pointer = static_cast<std::shared_ptr<T>*>(ptr);
      return *pointer;
      }
    return nullptr;
  }

  void RegisterFunctionWithLua(lua_State *L, int(func)(lua_State *), std::string type_name)
  {
    lua_pushcfunction(L, func);
    lua_setglobal(L, type_name.c_str());    
  }

  int LuaLoadLine(lua_State *L, std::string call)
  {
    int error = luaL_loadbuffer(L, call.c_str(), strlen(call.c_str()), "line");
    if (error) {
      fprintf(stderr, "%s", lua_tostring(L, -1));
      lua_pop(L, 1);  /* pop error message from the stack */
    }
    return error;
  }

  int LuaLoadFile(lua_State *L, std::string file)
  {
    // load in the underlying reader
    int error = luaL_loadfile(L, file.c_str());
    if (error) {
      fprintf(stderr, "%s", lua_tostring(L, -1));
      lua_pop(L, 1);  /* pop error message from the stack */
    }

    return error;
  }
  
  int ExecuteLua(lua_State *L)
  {
    int error = lua_pcall(L, 0, 0, 0);
    if (error) {
      fprintf(stderr, "%s", lua_tostring(L, -1));
      lua_pop(L, 1);  /* pop error message from the stack */
    }
    
    return error;
  }

  int ExecuteLuaLine(lua_State *L, std::string line)
  {
    return LuaLoadLine(L, line) || ExecuteLua(L);
  }
  
  int ExecuteLuaFile(lua_State *L, std::string file)
  {
    return LuaLoadFile(L, file) || ExecuteLua(L);
  }

  void interactive_loop(lua_State *L)
  {
    char buff[256];
    int error;
      while (fgets(buff, sizeof(buff), stdin) != NULL) {
	error = ExecuteLuaLine(L, std::string(buff));
        if (error) {
          fprintf(stderr, "%s", lua_tostring(L, -1));
          lua_pop(L, 1);  /* pop error message from the stack */
        }
      }
  }
}


/// This returns a std::shared_ptr<ParMesh> to Lua
/// Lua parameters
/// MeshReader((string)filename, (int)serial, (int)parallel)
static int MeshReader (lua_State *L)
{
  
  std::string relative_mesh_file = luaL_checkstring(L, 1);
  int serial = luaL_checkinteger(L, 2);
  int parallel = luaL_checkinteger(L, 3);
  
  // Open the mesh
  std::string mesh_file = std::string(SERAC_SRC_DIR) + relative_mesh_file;
  std::fstream imesh(mesh_file.c_str());
  auto mesh = std::make_unique<mfem::Mesh>(imesh, 1, 1, true);
  imesh.close();

  // Refine in serial
  mesh->UniformRefinement();

  // Initialize the parallel mesh and delete the serial mesh
  auto pmesh = std::make_shared<mfem::ParMesh>(MPI_COMM_WORLD, *mesh);

  // Refine the parallel mesh
  pmesh->UniformRefinement();

  int lua_return = luautil::CreateLuaObject<mfem::ParMesh>(L, pmesh, "ParMesh");
  std::cout << "Created the Mesh: " << mesh_file << " with options: {serial = " << serial << ",parallel = " << parallel << "}" << std::endl;
  
  return lua_return;
}

/// get the number of elements in the given mesh
int GetGlobalNE(lua_State * L)
{
  auto pmesh = luautil::GetPointer<mfem::ParMesh>(L, 1, "ParMesh");

  if (pmesh)
    {
      lua_pushnumber(L, pmesh->GetGlobalNE());
      return 1;
    }
  return 0;
}


/// Build the ThermalSolver
std::shared_ptr<ThermalSolver> ThermalSolverFactory(int order,
						    std::shared_ptr<mfem::ParMesh> pmesh,
						    TimestepMethod method,
						    std::shared_ptr<mfem::FunctionCoefficient> u_0,
						    std::shared_ptr<mfem::Coefficient> kappa,
						    double rel_tol,
						    double abs_tol,
						    int print_level,
						    int max_iter)
{

  // Initialize the second order thermal solver on the parallel mesh
  auto therm_solver = std::make_shared<ThermalSolver>(order, pmesh);

  // Set the time integration method
  therm_solver->SetTimestepper(method);

  // Initialize the temperature boundary condition
  std::vector<int> temp_bdr(pmesh->bdr_attributes.Max(), 1);

  // Set the temperature BC in the thermal solver
  therm_solver->SetTemperatureBCs(temp_bdr, u_0);

  // Set the conductivity of the thermal operator
  therm_solver->SetConductivity(kappa);

  // Define the linear solver params
  LinearSolverParameters params;
  params.rel_tol     = rel_tol;
  params.abs_tol     = abs_tol;
  params.print_level = print_level;
  params.max_iter    = max_iter;
  therm_solver->SetLinearSolverParameters(params);

  // Complete the setup without allocating the mass matrices and dynamic
  // operator
  therm_solver->CompleteSetup();

  return therm_solver;
}

double BoundaryTemperature(const mfem::Vector& x) { return x.Norml2(); }

/// lua method to build a thermalsolver
static int ThermalSolverBuilder(lua_State *L)
{
  int order = luaL_checkinteger(L,1);
  auto pmesh = luautil::GetPointer<mfem::ParMesh>(L, 2, "ParMesh");

  std::string timestepper = luaL_checkstring(L, 3);
  
  TimestepMethod method = TimestepMethod::QuasiStatic;
  MFEM_VERIFY(timestepper == "quasistatic", "Only Quasistatic has been implemented");

  auto u_0 = std::make_shared<mfem::FunctionCoefficient>(BoundaryTemperature);
  auto kappa = std::make_shared<mfem::ConstantCoefficient>(0.5);

  double rel_tol = luaL_checknumber(L,6);
  double abs_tol = luaL_checknumber(L,7);
  int print_level = luaL_checkinteger(L, 8);
  int max_iter = luaL_checkinteger(L, 9);
  
  auto thermal_solver = ThermalSolverFactory(order,
					     pmesh,
					     method,
					     u_0,
					     kappa,
					     rel_tol,
					     abs_tol,
					     print_level,
					     max_iter);
  
  int lua_return = luautil::CreateLuaObject<ThermalSolver>(L, thermal_solver, "ThermalSolver");
  return lua_return;
}

// increase by one time step
// Lua: Step(solver, dt)
static int ThermalSolverStep(lua_State *L)
{
  auto solver = luautil::GetPointer<ThermalSolver>(L, 1, "ThermalSolver");
  double dt = luaL_checknumber(L, 2);

  solver->AdvanceTimestep(dt);
  return 0;
}

/*
Lua commands can be found in /tests/serac_lua_deck.lua
*/
TEST(lua_test, static_solve)
{

  MPI_Barrier(MPI_COMM_WORLD);
  lua_State *L = luaL_newstate();   /* opens Lua */
  luaL_openlibs(L);               

  // register the new ParMesh metatable
  luautil::RegisterLuaTypes(L, {"ParMesh", "ThermalSolver"});

  luautil::RegisterFunctionWithLua(L, MeshReader, "MeshReader");
  luautil::RegisterFunctionWithLua(L, GetGlobalNE, "GetGlobalNE");    
  luautil::RegisterFunctionWithLua(L, ThermalSolverBuilder, "ThermalSolverBuilder");
  luautil::RegisterFunctionWithLua(L, ThermalSolverStep, "ThermalSolverStep");    
  
  // load in the underlying reader
  int error = luautil::ExecuteLuaFile(L, std::string(SERAC_SRC_DIR) + "/tests/serac_lua_test_reader.lua");

  std::cout << "Underlying reader loaded" << std::endl;
  
  error = luautil::ExecuteLuaFile(L, std::string(SERAC_SRC_DIR) + "/tests/serac_lua_deck.lua");

  // Call read_table automatically
  error = luautil::ExecuteLuaLine(L, "run_table(thermal_solver)");

  std::cout << "Trying to get the solver state" << std::endl;
  
  // grab the global lua "solver" variable
  lua_getglobal(L, "solver");
  auto therm_solver = luautil::GetPointer<ThermalSolver>(L, -1, "ThermalSolver");
  // Get the state grid function
  if (therm_solver) {
    auto state = therm_solver->GetState();
    std::cout << "Got the state" << std::endl;
  
    // Measure the L2 norm of the solution and check the value
    mfem::ConstantCoefficient zero(0.0);
    double                    u_norm = state[0].gf->ComputeLpError(2.0, zero);
    EXPECT_NEAR(2.56980679, u_norm, 0.00001);

  }
  lua_close(L);
}

int main(int argc, char* argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  MPI_Init(&argc, &argv);

  result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}
