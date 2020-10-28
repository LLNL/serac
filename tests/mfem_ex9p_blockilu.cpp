//                       MFEM Example 9 - Parallel Version
//
// Compile with: make ex9p
//
// Sample runs:
//    mpirun -np 4 ex9p -m ../data/periodic-segment.mesh -p 0 -dt 0.005
//    mpirun -np 4 ex9p -m ../data/periodic-square.mesh -p 0 -dt 0.01
//    mpirun -np 4 ex9p -m ../data/periodic-hexagon.mesh -p 0 -dt 0.01
//    mpirun -np 4 ex9p -m ../data/periodic-square.mesh -p 1 -dt 0.005 -tf 9
//    mpirun -np 4 ex9p -m ../data/periodic-hexagon.mesh -p 1 -dt 0.005 -tf 9
//    mpirun -np 4 ex9p -m ../data/amr-quad.mesh -p 1 -rp 1 -dt 0.002 -tf 9
//    mpirun -np 4 ex9p -m ../data/amr-quad.mesh -p 1 -rp 1 -dt 0.02 -s 13 -tf 9
//    mpirun -np 4 ex9p -m ../data/star-q3.mesh -p 1 -rp 1 -dt 0.004 -tf 9
//    mpirun -np 4 ex9p -m ../data/star-mixed.mesh -p 1 -rp 1 -dt 0.004 -tf 9
//    mpirun -np 4 ex9p -m ../data/disc-nurbs.mesh -p 1 -rp 1 -dt 0.005 -tf 9
//    mpirun -np 4 ex9p -m ../data/disc-nurbs.mesh -p 2 -rp 1 -dt 0.005 -tf 9
//    mpirun -np 4 ex9p -m ../data/periodic-square.mesh -p 3 -rp 2 -dt 0.0025 -tf 9 -vs 20
//    mpirun -np 4 ex9p -m ../data/periodic-cube.mesh -p 0 -o 2 -rp 1 -dt 0.01 -tf 8
//    mpirun -np 3 ex9p -m ../data/amr-hex.mesh -p 1 -rs 1 -rp 0 -dt 0.005 -tf 0.5
//
// Device sample runs:
//    mpirun -np 4 ex9p -pa
//    mpirun -np 4 ex9p -ea
//    mpirun -np 4 ex9p -fa
//    mpirun -np 4 ex9p -pa -m ../data/periodic-cube.mesh
//    mpirun -np 4 ex9p -pa -m ../data/periodic-cube.mesh -d cuda
//    mpirun -np 4 ex9p -ea -m ../data/periodic-cube.mesh -d cuda
//    mpirun -np 4 ex9p -fa -m ../data/periodic-cube.mesh -d cuda
//
// Description:  This example code solves the time-dependent advection equation
//               du/dt + v.grad(u) = 0, where v is a given fluid velocity, and
//               u0(x)=u(0,x) is a given initial condition.
//
//               The example demonstrates the use of Discontinuous Galerkin (DG)
//               bilinear forms in MFEM (face integrators), the use of implicit
//               and explicit ODE time integrators, the definition of periodic
//               boundary conditions through periodic meshes, as well as the use
//               of GLVis for persistent visualization of a time-evolving
//               solution. Saving of time-dependent data files for visualization
//               with VisIt (visit.llnl.gov) and ParaView (paraview.org), as
//               well as the optional saving with ADIOS2 (adios2.readthedocs.io)
//               are also illustrated.

#include <gtest/gtest.h>

#include <array>
#include <fstream>
#include <iostream>

#include "mfem.hpp"
#include "physics/utilities/equation_solver.hpp"
#include "serac_config.hpp"

using namespace std;
using namespace mfem;

// Solution obtained with unmodified ex9p
const std::array<double, 200> UNMODIFIED_SOLN = {
    28.894018,     28.906711,     28.883385,     28.927131,     28.874005,     28.927961,     28.884509,
    28.908173,     28.906884,     28.880166,     28.902028,     28.823792,     28.782051,     28.630917,
    28.434766,     28.190874,     27.845584,     27.514109,     27.113787,     26.694695,     26.294591,
    25.863878,     25.429219,     24.992041,     24.548686,     24.085625,     23.602508,     23.096184,
    22.552562,     21.949589,     21.286439,     20.557508,     19.757438,     18.882249,     17.949801,
    16.969856,     15.951942,     14.901164,     13.835909,     12.758287,     11.67046,      10.568555,
    9.4633545,     8.3498396,     7.2293048,     6.098465,      4.9800308,     3.8839091,     2.8487261,
    1.9016427,     1.0957054,     0.4838424,     0.11461909,    0.044595622,   0.01875509,    0.011029141,
    0.0041591384,  0.0023497672,  0.00077332336, 0.00045036114, 0.00012592644, 8.0257848e-05, 2.0519723e-05,
    1.2687743e-05, 3.7115821e-06, 1.7069413e-06, 6.7553687e-07, 1.9863657e-07, 1.0757026e-07, 2.4729792e-08,
    1.4168506e-08, 4.101342e-09,  1.510714e-09,  6.5701372e-10, 1.4894938e-10, 8.5009983e-11, 2.1070874e-11,
    8.6311499e-12, 3.3023731e-12, 7.5669102e-13, 4.1487402e-13, 9.5532683e-14, 4.0051918e-14, 1.4477674e-14,
    3.2615674e-15, 1.7446583e-15, 3.9033004e-16, 1.5943593e-16, 5.7034386e-17, 1.2323901e-17, 6.5383871e-18,
    1.4638441e-18, 5.6357417e-19, 2.0578303e-19, 4.214549e-20,  2.2314845e-20, 5.1179298e-21, 1.8110308e-21,
    6.8909427e-22, 1.3402368e-22, 7.041499e-23,  1.6868893e-23, 5.3817608e-24, 2.1628418e-24, 4.052847e-25,
    2.077315e-25,  5.2805297e-26, 1.4991756e-26, 6.4111195e-27, 1.1854418e-27, 5.7778698e-28, 1.5775273e-28,
    3.9627689e-29, 1.8055132e-29, 3.3888661e-30, 1.5252605e-30, 4.5133193e-31, 1.0063682e-31, 4.8541489e-32,
    9.5035079e-33, 3.8425014e-33, 1.2400856e-33, 2.4894274e-34, 1.2507591e-34, 2.6116985e-35, 9.2835404e-36,
    3.2802633e-36, 6.0873229e-37, 3.098741e-37,  7.0132001e-38, 2.1618231e-38, 8.3720174e-39, 1.490541e-39,
    7.4015334e-40, 1.8352718e-40, 4.8812576e-41, 2.0657884e-41, 3.6793218e-42, 1.7084339e-42, 4.6726625e-43,
    1.0773377e-43, 4.9368504e-44, 9.1410751e-45, 3.8190981e-45, 1.1567814e-45, 2.3503157e-46, 1.1444566e-46,
    2.2689232e-47, 8.2874588e-48, 2.7850172e-48, 5.1392687e-49, 2.5770285e-49, 5.5785421e-50, 1.7509667e-50,
    6.5244954e-51, 1.1413835e-51, 5.6431335e-52, 1.3495747e-52, 3.6180326e-53, 1.4884808e-53, 2.5915469e-54,
    1.2030245e-54, 3.1993179e-55, 7.3638427e-56, 3.3095322e-56, 5.9941373e-57, 2.4996726e-57, 7.4165183e-58,
    1.4925394e-58, 7.1770271e-59, 1.3975194e-59, 5.0699697e-60, 1.6799263e-60, 3.0561595e-61, 1.5190354e-61,
    3.2472367e-62, 1.0062134e-62, 3.7179698e-63, 6.4089481e-64, 3.1397626e-64, 7.4558905e-65, 1.9623254e-65,
    8.0421621e-66, 1.384426e-66,  6.3415152e-67, 1.6830301e-67, 3.7882438e-68, 1.7008593e-68, 3.0631187e-69,
    1.2525424e-69, 3.7252989e-70, 7.3241056e-71, 3.518606e-71,  6.8539413e-72, 2.4224178e-72, 8.0767717e-73,
    1.4398472e-73, 7.1226035e-74, 1.5313732e-74, 4.5983623e-75,
};

// Choice for the problem setup. The fluid velocity, initial condition and
// inflow boundary condition are chosen based on this parameter.
int problem;

// Velocity coefficient
void velocity_function(const Vector& x, Vector& v);

// Initial condition
double u0_function(const Vector& x);

// Inflow boundary condition
double inflow_function(const Vector& x);

// Mesh bounding box
Vector bb_min, bb_max;

class DG_Solver : public Solver {
private:
  HypreParMatrix &      M, &K;
  SparseMatrix          M_diag;
  HypreParMatrix*       A;
  serac::EquationSolver linear_solver;
  double                dt;

public:
  DG_Solver(HypreParMatrix& M_, HypreParMatrix& K_, const FiniteElementSpace& fes)
      : M(M_),
        K(K_),
        A(NULL),
        linear_solver(M.GetComm(),
                      serac::IterativeSolverParameters{.rel_tol     = 1e-9,
                                                       .abs_tol     = 0.0,
                                                       .print_level = 0,
                                                       .max_iter    = 100,
                                                       .lin_solver  = serac::LinearSolver::GMRES,
                                                       .prec        = serac::BlockILUPrec{fes.GetFE(0)->GetDof()}}),
        dt(-1.0)
  {
    linear_solver.linearSolver().iterative_mode = false;

    M.GetDiag(M_diag);
  }

  void SetTimeStep(double dt_)
  {
    if (dt_ != dt) {
      dt = dt_;
      // Form operator A = M - dt*K
      delete A;
      A = Add(-dt, K, 0.0, K);
      SparseMatrix A_diag;
      A->GetDiag(A_diag);
      A_diag.Add(1.0, M_diag);
      // this will also call SetOperator on the preconditioner
      linear_solver.SetOperator(*A);
    }
  }

  void SetOperator(const Operator& op) { linear_solver.SetOperator(op); }

  virtual void Mult(const Vector& x, Vector& y) const { linear_solver.Mult(x, y); }

  ~DG_Solver() { delete A; }
};

/** A time-dependent operator for the right-hand side of the ODE. The DG weak
    form of du/dt = -v.grad(u) is M du/dt = K u + b, where M and K are the mass
    and advection matrices, and b describes the flow on the boundary. This can
    be written as a general ODE, du/dt = M^{-1} (K u + b), and this class is
    used to evaluate the right-hand side. */
class FE_Evolution : public TimeDependentOperator {
private:
  OperatorHandle M, K;
  const Vector&  b;
  Solver*        M_prec;
  CGSolver       M_solver;
  DG_Solver*     dg_solver;

  mutable Vector z;

public:
  FE_Evolution(ParBilinearForm& _M, ParBilinearForm& _K, const Vector& _b);

  virtual void Mult(const Vector& x, Vector& y) const;
  virtual void ImplicitSolve(const double dt, const Vector& x, Vector& k);

  virtual ~FE_Evolution();
};

int main(int argc, char* argv[])
{
  // 1. Initialize MPI.
  int num_procs, myid;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  // 2. Parse command-line options.
  problem                     = 0;
  const char* mesh_file       = SERAC_REPO_DIR "/data/meshes/star.mesh";
  int         ser_ref_levels  = 2;
  int         par_ref_levels  = 0;
  int         order           = 3;
  bool        pa              = false;
  bool        ea              = false;
  bool        fa              = false;
  const char* device_config   = "cpu";
  int         ode_solver_type = 4;
  double      t_final         = 10.0;
  double      dt              = 0.01;
  bool        visualization   = true;
  bool        visit           = false;
  bool        paraview        = false;
  bool        adios2          = false;
  bool        binary          = false;
  int         vis_steps       = 5;

  int precision = 8;
  cout.precision(precision);

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&problem, "-p", "--problem", "Problem setup to use. See options in velocity_function().");
  args.AddOption(&ser_ref_levels, "-rs", "--refine-serial", "Number of times to refine the mesh uniformly in serial.");
  args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                 "Number of times to refine the mesh uniformly in parallel.");
  args.AddOption(&order, "-o", "--order", "Order (degree) of the finite elements.");
  args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa", "--no-partial-assembly", "Enable Partial Assembly.");
  args.AddOption(&ea, "-ea", "--element-assembly", "-no-ea", "--no-element-assembly", "Enable Element Assembly.");
  args.AddOption(&fa, "-fa", "--full-assembly", "-no-fa", "--no-full-assembly", "Enable Full Assembly.");
  args.AddOption(&device_config, "-d", "--device", "Device configuration string, see Device::Configure().");
  args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                 "ODE solver: 1 - Forward Euler,\n\t"
                 "            2 - RK2 SSP, 3 - RK3 SSP, 4 - RK4, 6 - RK6,\n\t"
                 "            11 - Backward Euler,\n\t"
                 "            12 - SDIRK23 (L-stable), 13 - SDIRK33,\n\t"
                 "            22 - Implicit Midpoint Method,\n\t"
                 "            23 - SDIRK23 (A-stable), 24 - SDIRK34");
  args.AddOption(&t_final, "-tf", "--t-final", "Final time; start time is 0.");
  args.AddOption(&dt, "-dt", "--time-step", "Time step.");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis", "--no-visualization",
                 "Enable or disable GLVis visualization.");
  args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit", "--no-visit-datafiles",
                 "Save data files for VisIt (visit.llnl.gov) visualization.");
  args.AddOption(&paraview, "-paraview", "--paraview-datafiles", "-no-paraview", "--no-paraview-datafiles",
                 "Save data files for ParaView (paraview.org) visualization.");
  args.AddOption(&adios2, "-adios2", "--adios2-streams", "-no-adios2", "--no-adios2-streams",
                 "Save data using adios2 streams.");
  args.AddOption(&binary, "-binary", "--binary-datafiles", "-ascii", "--ascii-datafiles",
                 "Use binary (Sidre) or ascii format for VisIt data files.");
  args.AddOption(&vis_steps, "-vs", "--visualization-steps", "Visualize every n-th timestep.");
  args.Parse();
  if (!args.Good()) {
    if (myid == 0) {
      args.PrintUsage(cout);
    }
    MPI_Finalize();
    return 1;
  }
  if (myid == 0) {
    args.PrintOptions(cout);
  }

  Device device(device_config);
  if (myid == 0) {
    device.Print();
  }

  // 3. Read the serial mesh from the given mesh file on all processors. We can
  //    handle geometrically periodic meshes in this code.
  Mesh* mesh = new Mesh(mesh_file, 1, 1);
  int   dim  = mesh->Dimension();

  // 4. Define the ODE solver used for time integration. Several explicit
  //    Runge-Kutta methods are available.
  ODESolver* ode_solver = NULL;
  switch (ode_solver_type) {
    // Explicit methods
    case 1:
      ode_solver = new ForwardEulerSolver;
      break;
    case 2:
      ode_solver = new RK2Solver(1.0);
      break;
    case 3:
      ode_solver = new RK3SSPSolver;
      break;
    case 4:
      ode_solver = new RK4Solver;
      break;
    case 6:
      ode_solver = new RK6Solver;
      break;
    // Implicit (L-stable) methods
    case 11:
      ode_solver = new BackwardEulerSolver;
      break;
    case 12:
      ode_solver = new SDIRK23Solver(2);
      break;
    case 13:
      ode_solver = new SDIRK33Solver;
      break;
    // Implicit A-stable methods (not L-stable)
    case 22:
      ode_solver = new ImplicitMidpointSolver;
      break;
    case 23:
      ode_solver = new SDIRK23Solver;
      break;
    case 24:
      ode_solver = new SDIRK34Solver;
      break;
    default:
      if (myid == 0) {
        cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
      }
      delete mesh;
      MPI_Finalize();
      return 3;
  }

  // 5. Refine the mesh in serial to increase the resolution. In this example
  //    we do 'ser_ref_levels' of uniform refinement, where 'ser_ref_levels' is
  //    a command-line parameter. If the mesh is of NURBS type, we convert it
  //    to a (piecewise-polynomial) high-order mesh.
  for (int lev = 0; lev < ser_ref_levels; lev++) {
    mesh->UniformRefinement();
  }
  if (mesh->NURBSext) {
    mesh->SetCurvature(max(order, 1));
  }
  mesh->GetBoundingBox(bb_min, bb_max, max(order, 1));

  // 6. Define the parallel mesh by a partitioning of the serial mesh. Refine
  //    this mesh further in parallel to increase the resolution. Once the
  //    parallel mesh is defined, the serial mesh can be deleted.
  ParMesh* pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
  delete mesh;
  for (int lev = 0; lev < par_ref_levels; lev++) {
    pmesh->UniformRefinement();
  }

  // 7. Define the parallel discontinuous DG finite element space on the
  //    parallel refined mesh of the given polynomial order.
  DG_FECollection        fec(order, dim, BasisType::GaussLobatto);
  ParFiniteElementSpace* fes = new ParFiniteElementSpace(pmesh, &fec);

  HYPRE_Int global_vSize = fes->GlobalTrueVSize();
  if (myid == 0) {
    cout << "Number of unknowns: " << global_vSize << endl;
  }

  // 8. Set up and assemble the parallel bilinear and linear forms (and the
  //    parallel hypre matrices) corresponding to the DG discretization. The
  //    DGTraceIntegrator involves integrals over mesh interior faces.
  VectorFunctionCoefficient velocity(dim, velocity_function);
  FunctionCoefficient       inflow(inflow_function);
  FunctionCoefficient       u0(u0_function);

  ParBilinearForm* m = new ParBilinearForm(fes);
  ParBilinearForm* k = new ParBilinearForm(fes);
  if (pa) {
    m->SetAssemblyLevel(AssemblyLevel::PARTIAL);
    k->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  } else if (ea) {
    m->SetAssemblyLevel(AssemblyLevel::ELEMENT);
    k->SetAssemblyLevel(AssemblyLevel::ELEMENT);
  } else if (fa) {
    m->SetAssemblyLevel(AssemblyLevel::FULL);
    k->SetAssemblyLevel(AssemblyLevel::FULL);
  }

  m->AddDomainIntegrator(new MassIntegrator);
  k->AddDomainIntegrator(new ConvectionIntegrator(velocity, -1.0));
  k->AddInteriorFaceIntegrator(new TransposeIntegrator(new DGTraceIntegrator(velocity, 1.0, -0.5)));
  k->AddBdrFaceIntegrator(new TransposeIntegrator(new DGTraceIntegrator(velocity, 1.0, -0.5)));

  ParLinearForm* b = new ParLinearForm(fes);
  b->AddBdrFaceIntegrator(new BoundaryFlowIntegrator(inflow, velocity, -1.0, -0.5));

  int skip_zeros = 0;
  m->Assemble();
  k->Assemble(skip_zeros);
  b->Assemble();
  m->Finalize();
  k->Finalize(skip_zeros);

  HypreParVector* B = b->ParallelAssemble();

  // 9. Define the initial conditions, save the corresponding grid function to
  //    a file and (optionally) save data in the VisIt format and initialize
  //    GLVis visualization.
  ParGridFunction* u = new ParGridFunction(fes);
  u->ProjectCoefficient(u0);
  HypreParVector* U = u->GetTrueDofs();

  {
    ostringstream mesh_name, sol_name;
    mesh_name << "ex9-mesh." << setfill('0') << setw(6) << myid;
    sol_name << "ex9-init." << setfill('0') << setw(6) << myid;
    ofstream omesh(mesh_name.str().c_str());
    omesh.precision(precision);
    pmesh->Print(omesh);
    ofstream osol(sol_name.str().c_str());
    osol.precision(precision);
    u->Save(osol);
  }

  // Create data collection for solution output: either VisItDataCollection for
  // ascii data files, or SidreDataCollection for binary data files.
  DataCollection* dc = NULL;
  if (visit) {
    if (binary) {
#ifdef MFEM_USE_SIDRE
      dc = new SidreDataCollection("Example9-Parallel", pmesh);
#else
      MFEM_ABORT("Must build with MFEM_USE_SIDRE=YES for binary output.");
#endif
    } else {
      dc = new VisItDataCollection("Example9-Parallel", pmesh);
      dc->SetPrecision(precision);
      // To save the mesh using MFEM's parallel mesh format:
      // dc->SetFormat(DataCollection::PARALLEL_FORMAT);
    }
    dc->RegisterField("solution", u);
    dc->SetCycle(0);
    dc->SetTime(0.0);
    dc->Save();
  }

  ParaViewDataCollection* pd = NULL;
  if (paraview) {
    pd = new ParaViewDataCollection("Example9P", pmesh);
    pd->SetPrefixPath("ParaView");
    pd->RegisterField("solution", u);
    pd->SetLevelsOfDetail(order);
    pd->SetDataFormat(VTKFormat::BINARY);
    pd->SetHighOrderOutput(true);
    pd->SetCycle(0);
    pd->SetTime(0.0);
    pd->Save();
  }

  // Optionally output a BP (binary pack) file using ADIOS2. This can be
  // visualized with the ParaView VTX reader.
#ifdef MFEM_USE_ADIOS2
  ADIOS2DataCollection* adios2_dc = NULL;
  if (adios2) {
    std::string postfix(mesh_file);
    postfix.erase(0, std::string("../data/").size());
    postfix += "_o" + std::to_string(order);
    const std::string collection_name = "ex9-p-" + postfix + ".bp";

    adios2_dc = new ADIOS2DataCollection(MPI_COMM_WORLD, collection_name, pmesh);
    // output data substreams are half the number of mpi processes
    adios2_dc->SetParameter("SubStreams", std::to_string(num_procs / 2));
    // adios2_dc->SetLevelsOfDetail(2);
    adios2_dc->RegisterField("solution", u);
    adios2_dc->SetCycle(0);
    adios2_dc->SetTime(0.0);
    adios2_dc->Save();
  }
#endif

  socketstream sout;
  if (visualization) {
    char vishost[] = "localhost";
    int  visport   = 19916;
    sout.open(vishost, visport);
    if (!sout) {
      if (myid == 0) cout << "Unable to connect to GLVis server at " << vishost << ':' << visport << endl;
      visualization = false;
      if (myid == 0) {
        cout << "GLVis visualization disabled.\n";
      }
    } else {
      sout << "parallel " << num_procs << " " << myid << "\n";
      sout.precision(precision);
      sout << "solution\n" << *pmesh << *u;
      sout << "pause\n";
      sout << flush;
      if (myid == 0)
        cout << "GLVis visualization paused."
             << " Press space (in the GLVis window) to resume it.\n";
    }
  }

  // 10. Define the time-dependent evolution operator describing the ODE
  //     right-hand side, and perform time-integration (looping over the time
  //     iterations, ti, with a time-step dt).
  FE_Evolution adv(*m, *k, *B);

  double t = 0.0;
  adv.SetTime(t);
  ode_solver->Init(adv);

  bool done = false;
  for (int ti = 0; !done;) {
    double dt_real = min(dt, t_final - t);
    ode_solver->Step(*U, t, dt_real);
    ti++;

    done = (t >= t_final - 1e-8 * dt);

    if (done || ti % vis_steps == 0) {
      if (myid == 0) {
        cout << "time step: " << ti << ", time: " << t << endl;
      }

      // 11. Extract the parallel grid function corresponding to the finite
      //     element approximation U (the local solution on each processor).
      *u = *U;

      if (myid == 0) {
        EXPECT_FLOAT_EQ(u->Norml2(), UNMODIFIED_SOLN.at((ti / vis_steps) - 1));
      }

      if (visualization) {
        sout << "parallel " << num_procs << " " << myid << "\n";
        sout << "solution\n" << *pmesh << *u << flush;
      }

      if (visit) {
        dc->SetCycle(ti);
        dc->SetTime(t);
        dc->Save();
      }

      if (paraview) {
        pd->SetCycle(ti);
        pd->SetTime(t);
        pd->Save();
      }

#ifdef MFEM_USE_ADIOS2
      // transient solutions can be visualized with ParaView
      if (adios2) {
        adios2_dc->SetCycle(ti);
        adios2_dc->SetTime(t);
        adios2_dc->Save();
      }
#endif
    }
  }

  // 12. Save the final solution in parallel. This output can be viewed later
  //     using GLVis: "glvis -np <np> -m ex9-mesh -g ex9-final".
  {
    *u = *U;
    ostringstream sol_name;
    sol_name << "ex9-final." << setfill('0') << setw(6) << myid;
    ofstream osol(sol_name.str().c_str());
    osol.precision(precision);
    u->Save(osol);
  }

  // 13. Free the used memory.
  delete U;
  delete u;
  delete B;
  delete b;
  delete k;
  delete m;
  delete fes;
  delete pmesh;
  delete ode_solver;
  delete pd;
#ifdef MFEM_USE_ADIOS2
  if (adios2) {
    delete adios2_dc;
  }
#endif
  delete dc;

  MPI_Finalize();
  return 0;
}

// Implementation of class FE_Evolution
FE_Evolution::FE_Evolution(ParBilinearForm& _M, ParBilinearForm& _K, const Vector& _b)
    : TimeDependentOperator(_M.Height()), b(_b), M_solver(_M.ParFESpace()->GetComm()), z(_M.Height())
{
  if (_M.GetAssemblyLevel() == AssemblyLevel::LEGACYFULL) {
    M.Reset(_M.ParallelAssemble(), true);
    K.Reset(_K.ParallelAssemble(), true);
  } else {
    M.Reset(&_M, false);
    K.Reset(&_K, false);
  }

  M_solver.SetOperator(*M);

  Array<int> ess_tdof_list;
  if (_M.GetAssemblyLevel() == AssemblyLevel::LEGACYFULL) {
    HypreParMatrix& M_mat      = *M.As<HypreParMatrix>();
    HypreParMatrix& K_mat      = *K.As<HypreParMatrix>();
    HypreSmoother*  hypre_prec = new HypreSmoother(M_mat, HypreSmoother::Jacobi);
    M_prec                     = hypre_prec;

    dg_solver = new DG_Solver(M_mat, K_mat, *_M.FESpace());
  } else {
    M_prec    = new OperatorJacobiSmoother(_M, ess_tdof_list);
    dg_solver = NULL;
  }

  M_solver.SetPreconditioner(*M_prec);
  M_solver.iterative_mode = false;
  M_solver.SetRelTol(1e-9);
  M_solver.SetAbsTol(0.0);
  M_solver.SetMaxIter(100);
  M_solver.SetPrintLevel(0);
}

void FE_Evolution::ImplicitSolve(const double dt, const Vector& x, Vector& k)
{
  K->Mult(x, z);
  z += b;
  dg_solver->SetTimeStep(dt);
  dg_solver->Mult(z, k);
}

void FE_Evolution::Mult(const Vector& x, Vector& y) const
{
  // y = M^{-1} (K x + b)
  K->Mult(x, z);
  z += b;
  M_solver.Mult(z, y);
}

FE_Evolution::~FE_Evolution()
{
  delete M_prec;
  delete dg_solver;
}

// Velocity coefficient
void velocity_function(const Vector& x, Vector& v)
{
  int dim = x.Size();

  // map to the reference [-1,1] domain
  Vector X(dim);
  for (int i = 0; i < dim; i++) {
    double center = (bb_min[i] + bb_max[i]) * 0.5;
    X(i)          = 2 * (x(i) - center) / (bb_max[i] - bb_min[i]);
  }

  switch (problem) {
    case 0: {
      // Translations in 1D, 2D, and 3D
      switch (dim) {
        case 1:
          v(0) = 1.0;
          break;
        case 2:
          v(0) = sqrt(2. / 3.);
          v(1) = sqrt(1. / 3.);
          break;
        case 3:
          v(0) = sqrt(3. / 6.);
          v(1) = sqrt(2. / 6.);
          v(2) = sqrt(1. / 6.);
          break;
      }
      break;
    }
    case 1:
    case 2: {
      // Clockwise rotation in 2D around the origin
      const double w = M_PI / 2;
      switch (dim) {
        case 1:
          v(0) = 1.0;
          break;
        case 2:
          v(0) = w * X(1);
          v(1) = -w * X(0);
          break;
        case 3:
          v(0) = w * X(1);
          v(1) = -w * X(0);
          v(2) = 0.0;
          break;
      }
      break;
    }
    case 3: {
      // Clockwise twisting rotation in 2D around the origin
      const double w = M_PI / 2;
      double       d = max((X(0) + 1.) * (1. - X(0)), 0.) * max((X(1) + 1.) * (1. - X(1)), 0.);
      d              = d * d;
      switch (dim) {
        case 1:
          v(0) = 1.0;
          break;
        case 2:
          v(0) = d * w * X(1);
          v(1) = -d * w * X(0);
          break;
        case 3:
          v(0) = d * w * X(1);
          v(1) = -d * w * X(0);
          v(2) = 0.0;
          break;
      }
      break;
    }
  }
}

// Initial condition
double u0_function(const Vector& x)
{
  int dim = x.Size();

  // map to the reference [-1,1] domain
  Vector X(dim);
  for (int i = 0; i < dim; i++) {
    double center = (bb_min[i] + bb_max[i]) * 0.5;
    X(i)          = 2 * (x(i) - center) / (bb_max[i] - bb_min[i]);
  }

  // SERAC EDIT: removed the internal returns to quiet fallthrough warning
  double retValue = 0.0;
  switch (problem) {
    case 0:
      [[fallthrough]];
    case 1: {
      switch (dim) {
        case 1:
          retValue = exp(-40. * pow(X(0) - 0.5, 2));
          break;
        case 2:
          [[fallthrough]];
        case 3: {
          double rx = 0.45, ry = 0.25, cx = 0., cy = -0.2, w = 10.;
          if (dim == 3) {
            const double s = (1. + 0.25 * cos(2 * M_PI * X(2)));
            rx *= s;
            ry *= s;
          }
          retValue = (erfc(w * (X(0) - cx - rx)) * erfc(-w * (X(0) - cx + rx)) * erfc(w * (X(1) - cy - ry)) *
                  erfc(-w * (X(1) - cy + ry))) /
                 16;
          break;
        }
      }
      break;
    }
    case 2: {
      double x_ = X(0), y_ = X(1), rho, phi;
      rho = hypot(x_, y_);
      phi = atan2(y_, x_);
      retValue = pow(sin(M_PI * rho), 2) * sin(3 * phi);
      break;
    }
    case 3: {
      const double f = M_PI;
      retValue = sin(f * X(0)) * sin(f * X(1));
      break;
    }
  }
  return retValue;
}

// Inflow boundary condition (zero for the problems considered in this example)
double inflow_function(const Vector&)
{
  switch (problem) {
    case 0:
    case 1:
    case 2:
    case 3:
      return 0.0;
  }
  return 0.0;
}
