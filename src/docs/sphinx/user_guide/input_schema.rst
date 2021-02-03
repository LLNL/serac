=================
Input File Schema
=================

Below is the documentation for Serac input files, generated automatically by `Axom's inlet component <https://axom.readthedocs.io/en/develop/axom/inlet/docs/sphinx/index.html>`_.

.. |uncheck|    unicode:: U+2610 .. UNCHECKED BOX
.. |check|      unicode:: U+2611 .. CHECKED BOX

----------
input_file
----------

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - t_final
     - Final time for simulation.
     - 1.000000
     - 
     - |uncheck|
   * - dt
     - Time step.
     - 0.250000
     - 
     - |uncheck|

---------
main_mesh
---------

Description: The main mesh for the problem

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - mesh
     - Path to Mesh file
     - 
     - 
     - |check|
   * - ser_ref_levels
     - Number of times to refine the mesh uniformly in serial.
     - 0
     - 
     - |uncheck|
   * - par_ref_levels
     - Number of times to refine the mesh uniformly in parallel.
     - 0
     - 
     - |uncheck|

---------------
nonlinear_solid
---------------

Description: Finite deformation solid mechanics module

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - order
     - Order degree of the finite elements.
     - 1
     - 1 to 8
     - |uncheck|
   * - mu
     - Shear modulus in the Neo-Hookean hyperelastic model.
     - 0.250000
     - 
     - |uncheck|
   * - K
     - Bulk modulus in the Neo-Hookean hyperelastic model.
     - 5.000000
     - 
     - |uncheck|

----------------
stiffness_solver
----------------

Description: Linear and Nonlinear stiffness Solver Parameters.


------
linear
------

Description: Linear Equation Solver Parameters

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - type
     - The type of solver parameters to use (iterative|direct)
     - 
     - iterative, direct
     - |check|

-----------------
iterative_options
-----------------

Description: Iterative solver parameters

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - rel_tol
     - Relative tolerance for the linear solve.
     - 0.000001
     - 
     - |uncheck|
   * - abs_tol
     - Absolute tolerance for the linear solve.
     - 0.000000
     - 
     - |uncheck|
   * - max_iter
     - Maximum iterations for the linear solve.
     - 5000
     - 
     - |uncheck|
   * - print_level
     - Linear print level.
     - 0
     - 
     - |uncheck|
   * - solver_type
     - Solver type (gmres|minres).
     - gmres
     - 
     - |uncheck|
   * - prec_type
     - Preconditioner type (JacobiSmoother|L1JacobiSmoother|AMG|BlockILU).
     - JacobiSmoother
     - 
     - |uncheck|

--------------
direct_options
--------------

Description: Direct solver parameters

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - print_level
     - Linear print level.
     - 0
     - 
     - |uncheck|

---------
nonlinear
---------

Description: Newton Equation Solver Parameters

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - rel_tol
     - Relative tolerance for the Newton solve.
     - 0.010000
     - 
     - |uncheck|
   * - abs_tol
     - Absolute tolerance for the Newton solve.
     - 0.000100
     - 
     - |uncheck|
   * - max_iter
     - Maximum iterations for the Newton solve.
     - 500
     - 
     - |uncheck|
   * - print_level
     - Nonlinear print level.
     - 0
     - 
     - |uncheck|
   * - solver_type
     - Solver type (MFEMNewton|KINFullStep|KINLineSearch)
     - MFEMNewton
     - 
     - |uncheck|

--------
dynamics
--------

Description: Parameters for mass matrix inversion

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - timestepper
     - Timestepper (ODE) method to use
     - 
     - 
     - |uncheck|
   * - enforcement_method
     - Time-varying constraint enforcement method to use
     - 
     - 
     - |uncheck|

--------------
boundary_conds
--------------


----------------
_inlet_container
----------------

Description: Table of boundary conditions

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - _inlet_container_indices
     - 
     - 
     - 
     - |uncheck|

------------
displacement
------------

Description: Table of boundary conditions

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - vec_coef
     - The function to use for an mfem::VectorFunctionCoefficient
     - 
     - 
     - |uncheck|
   * - coef
     - The function to use for an mfem::FunctionCoefficient
     - 
     - 
     - |uncheck|
   * - component
     - The vector component to which the scalar coefficient should be applied
     - 
     - 
     - |uncheck|

-----
attrs
-----


----------------
_inlet_container
----------------

Description: Boundary attributes to which the BC should be applied

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - 1
     - 
     - 
     - 
     - |uncheck|

--------
traction
--------

Description: Table of boundary conditions

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - vec_coef
     - The function to use for an mfem::VectorFunctionCoefficient
     - 
     - 
     - |uncheck|
   * - coef
     - The function to use for an mfem::FunctionCoefficient
     - 
     - 
     - |uncheck|
   * - component
     - The vector component to which the scalar coefficient should be applied
     - 
     - 
     - |uncheck|

-----
attrs
-----


----------------
_inlet_container
----------------

Description: Boundary attributes to which the BC should be applied

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - 1
     - 
     - 
     - 
     - |uncheck|

--------------------
initial_displacement
--------------------

Description: Coefficient for initial condition

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - vec_coef
     - The function to use for an mfem::VectorFunctionCoefficient
     - 
     - 
     - |uncheck|
   * - coef
     - The function to use for an mfem::FunctionCoefficient
     - 
     - 
     - |uncheck|
   * - component
     - The vector component to which the scalar coefficient should be applied
     - 
     - 
     - |uncheck|

----------------
initial_velocity
----------------

Description: Coefficient for initial condition

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - vec_coef
     - The function to use for an mfem::VectorFunctionCoefficient
     - 
     - 
     - |uncheck|
   * - coef
     - The function to use for an mfem::FunctionCoefficient
     - 
     - 
     - |uncheck|
   * - component
     - The vector component to which the scalar coefficient should be applied
     - 
     - 
     - |uncheck|
