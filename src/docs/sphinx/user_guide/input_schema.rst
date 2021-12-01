.. ## Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
.. ## other Serac Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

====================
Input File Structure
====================

Below is the documentation for Serac input files, generated automatically by `Axom's inlet component <https://axom.readthedocs.io/en/develop/axom/inlet/docs/sphinx/index.html>`_.

.. |uncheck|    unicode:: U+2610 .. UNCHECKED BOX
.. |check|      unicode:: U+2611 .. CHECKED BOX

==================
Input file Options
==================

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
   * - output_type
     - Desired output format
     - VisIt
     - GLVis, ParaView, VisIt, SidreVisIt
     - |uncheck|
   * - dt
     - Time step.
     - 0.250000
     - 
     - |uncheck|
   * - t_final
     - Final time for simulation.
     - 1.000000
     - 
     - |uncheck|

-------------
thermal_solid
-------------

Description: Thermal solid module


----------------------
coef_thermal_expansion
----------------------

Description: Coefficient of thermal expansion for isotropic thermal expansion

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - constant
     - The constant scalar value to use as the coefficient
     - 
     - 
     - |uncheck|
   * - component
     - The vector component to which the scalar coefficient should be applied
     - 
     - 
     - |uncheck|
.. list-table:: Functions
   :widths: 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Function Name
     - Description
     - Signature
     - Required
   * - scalar_function
     - The function to use for an mfem::FunctionCoefficient
     - Double(Vector, Double)
     - |uncheck|
   * - vector_function
     - The function to use for an mfem::VectorFunctionCoefficient
     - Vector(Vector, Double)
     - |uncheck|

---------------
vector_constant
---------------

Description: The constant vector to use as the coefficient

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - z
     - z-component of vector
     - 
     - 
     - |uncheck|
   * - y
     - y-component of vector
     - 
     - 
     - |uncheck|
   * - x
     - x-component of vector
     - 
     - 
     - |uncheck|

---------------------
reference_temperature
---------------------

Description: Coefficient for the reference temperature for isotropic thermal expansion

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - constant
     - The constant scalar value to use as the coefficient
     - 
     - 
     - |uncheck|
   * - component
     - The vector component to which the scalar coefficient should be applied
     - 
     - 
     - |uncheck|
.. list-table:: Functions
   :widths: 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Function Name
     - Description
     - Signature
     - Required
   * - scalar_function
     - The function to use for an mfem::FunctionCoefficient
     - Double(Vector, Double)
     - |uncheck|
   * - vector_function
     - The function to use for an mfem::VectorFunctionCoefficient
     - Vector(Vector, Double)
     - |uncheck|

---------------
vector_constant
---------------

Description: The constant vector to use as the coefficient

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - z
     - z-component of vector
     - 
     - 
     - |uncheck|
   * - y
     - y-component of vector
     - 
     - 
     - |uncheck|
   * - x
     - x-component of vector
     - 
     - 
     - |uncheck|

------------------
thermal_conduction
------------------

Description: Thermal conduction module

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - rho
     - Density
     - 1.000000
     - 
     - |uncheck|
   * - cp
     - Specific heat capacity
     - 1.000000
     - 
     - |uncheck|
   * - kappa
     - Thermal conductivity
     - 0.500000
     - 
     - |uncheck|
   * - order
     - Order degree of the finite elements.
     - 1
     - 1 to 8
     - |uncheck|

--------------
boundary_conds
--------------


--------------------
Collection contents:
--------------------


------
source
------

Description: Scalar source term (RHS of the thermal conduction PDE)

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - constant
     - The constant scalar value to use as the coefficient
     - 
     - 
     - |uncheck|
   * - component
     - The vector component to which the scalar coefficient should be applied
     - 
     - 
     - |uncheck|
.. list-table:: Functions
   :widths: 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Function Name
     - Description
     - Signature
     - Required
   * - scalar_function
     - The function to use for an mfem::FunctionCoefficient
     - Double(Vector, Double)
     - |uncheck|
   * - vector_function
     - The function to use for an mfem::VectorFunctionCoefficient
     - Vector(Vector, Double)
     - |uncheck|

---------------
vector_constant
---------------

Description: The constant vector to use as the coefficient

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - z
     - z-component of vector
     - 
     - 
     - |uncheck|
   * - y
     - y-component of vector
     - 
     - 
     - |uncheck|
   * - x
     - x-component of vector
     - 
     - 
     - |uncheck|

-------------------
initial_temperature
-------------------

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
   * - constant
     - The constant scalar value to use as the coefficient
     - 
     - 
     - |uncheck|
   * - component
     - The vector component to which the scalar coefficient should be applied
     - 
     - 
     - |uncheck|
.. list-table:: Functions
   :widths: 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Function Name
     - Description
     - Signature
     - Required
   * - scalar_function
     - The function to use for an mfem::FunctionCoefficient
     - Double(Vector, Double)
     - |uncheck|
   * - vector_function
     - The function to use for an mfem::VectorFunctionCoefficient
     - Vector(Vector, Double)
     - |uncheck|

---------------
vector_constant
---------------

Description: The constant vector to use as the coefficient

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - y
     - y-component of vector
     - 
     - 
     - |uncheck|
   * - z
     - z-component of vector
     - 
     - 
     - |uncheck|
   * - x
     - x-component of vector
     - 
     - 
     - |uncheck|

------------------
nonlinear_reaction
------------------

Description: Nonlinear reaction term parameters

.. list-table:: Functions
   :widths: 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Function Name
     - Description
     - Signature
     - Required
   * - d_reaction_function
     - Derivative of the nonlinear reaction function dq = dq / dTemperature
     - Double(Double)
     - |uncheck|
   * - reaction_function
     - Nonlinear reaction function q = q(temperature)
     - Double(Double)
     - |uncheck|

-----
scale
-----

Description: Spatially varying scale factor for the reaction

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - constant
     - The constant scalar value to use as the coefficient
     - 
     - 
     - |uncheck|
   * - component
     - The vector component to which the scalar coefficient should be applied
     - 
     - 
     - |uncheck|
.. list-table:: Functions
   :widths: 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Function Name
     - Description
     - Signature
     - Required
   * - scalar_function
     - The function to use for an mfem::FunctionCoefficient
     - Double(Vector, Double)
     - |uncheck|
   * - vector_function
     - The function to use for an mfem::VectorFunctionCoefficient
     - Vector(Vector, Double)
     - |uncheck|

---------------
vector_constant
---------------

Description: The constant vector to use as the coefficient

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - z
     - z-component of vector
     - 
     - 
     - |uncheck|
   * - y
     - y-component of vector
     - 
     - 
     - |uncheck|
   * - x
     - x-component of vector
     - 
     - 
     - |uncheck|

---------------
equation_solver
---------------

Description: Linear and Nonlinear stiffness Solver Parameters.


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
   * - solver_type
     - Solver type (MFEMNewton|KINFullStep|KINLineSearch)
     - MFEMNewton
     - 
     - |uncheck|
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
   * - prec_type
     - Preconditioner type (JacobiSmoother|L1JacobiSmoother|AMG|BlockILU).
     - JacobiSmoother
     - 
     - |uncheck|
   * - solver_type
     - Solver type (gmres|minres|cg).
     - gmres
     - 
     - |uncheck|
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
   * - enforcement_method
     - Time-varying constraint enforcement method to use
     - 
     - 
     - |uncheck|
   * - timestepper
     - Timestepper (ODE) method to use
     - 
     - 
     - |uncheck|

-----
solid
-----

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
   * - viscosity
     - Viscosity constant
     - 0.000000
     - 
     - |uncheck|
   * - density
     - Initial mass density
     - 1.000000
     - 
     - |uncheck|
   * - material_nonlin
     - Flag to include material nonlinearities (linear elastic vs. neo-Hookean material model).
     - True
     - 
     - |uncheck|
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
   * - geometric_nonlin
     - Flag to include geometric nonlinearities in the residual calculation.
     - True
     - 
     - |uncheck|
   * - K
     - Bulk modulus in the Neo-Hookean hyperelastic model.
     - 5.000000
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
   * - constant
     - The constant scalar value to use as the coefficient
     - 
     - 
     - |uncheck|
   * - component
     - The vector component to which the scalar coefficient should be applied
     - 
     - 
     - |uncheck|
.. list-table:: Functions
   :widths: 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Function Name
     - Description
     - Signature
     - Required
   * - scalar_function
     - The function to use for an mfem::FunctionCoefficient
     - Double(Vector, Double)
     - |uncheck|
   * - vector_function
     - The function to use for an mfem::VectorFunctionCoefficient
     - Vector(Vector, Double)
     - |uncheck|

---------------
vector_constant
---------------

Description: The constant vector to use as the coefficient

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - z
     - z-component of vector
     - 
     - 
     - |uncheck|
   * - y
     - y-component of vector
     - 
     - 
     - |uncheck|
   * - x
     - x-component of vector
     - 
     - 
     - |uncheck|

---------------
equation_solver
---------------

Description: Linear and Nonlinear stiffness Solver Parameters.


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
   * - solver_type
     - Solver type (MFEMNewton|KINFullStep|KINLineSearch)
     - MFEMNewton
     - 
     - |uncheck|
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
   * - print_level
     - Nonlinear print level.
     - 0
     - 
     - |uncheck|
   * - max_iter
     - Maximum iterations for the Newton solve.
     - 500
     - 
     - |uncheck|

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
   * - prec_type
     - Preconditioner type (JacobiSmoother|L1JacobiSmoother|AMG|BlockILU).
     - JacobiSmoother
     - 
     - |uncheck|
   * - solver_type
     - Solver type (gmres|minres|cg).
     - gmres
     - 
     - |uncheck|
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

--------------
boundary_conds
--------------


--------------------
Collection contents:
--------------------


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
   * - enforcement_method
     - Time-varying constraint enforcement method to use
     - 
     - 
     - |uncheck|
   * - timestepper
     - Timestepper (ODE) method to use
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
   * - constant
     - The constant scalar value to use as the coefficient
     - 
     - 
     - |uncheck|
   * - component
     - The vector component to which the scalar coefficient should be applied
     - 
     - 
     - |uncheck|
.. list-table:: Functions
   :widths: 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Function Name
     - Description
     - Signature
     - Required
   * - scalar_function
     - The function to use for an mfem::FunctionCoefficient
     - Double(Vector, Double)
     - |uncheck|
   * - vector_function
     - The function to use for an mfem::VectorFunctionCoefficient
     - Vector(Vector, Double)
     - |uncheck|

---------------
vector_constant
---------------

Description: The constant vector to use as the coefficient

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - z
     - z-component of vector
     - 
     - 
     - |uncheck|
   * - y
     - y-component of vector
     - 
     - 
     - |uncheck|
   * - x
     - x-component of vector
     - 
     - 
     - |uncheck|

------------------
thermal_conduction
------------------

Description: Thermal conduction module

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - cp
     - Specific heat capacity
     - 1.000000
     - 
     - |uncheck|
   * - rho
     - Density
     - 1.000000
     - 
     - |uncheck|
   * - kappa
     - Thermal conductivity
     - 0.500000
     - 
     - |uncheck|
   * - order
     - Order degree of the finite elements.
     - 1
     - 1 to 8
     - |uncheck|

-------------------
initial_temperature
-------------------

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
   * - constant
     - The constant scalar value to use as the coefficient
     - 
     - 
     - |uncheck|
   * - component
     - The vector component to which the scalar coefficient should be applied
     - 
     - 
     - |uncheck|
.. list-table:: Functions
   :widths: 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Function Name
     - Description
     - Signature
     - Required
   * - scalar_function
     - The function to use for an mfem::FunctionCoefficient
     - Double(Vector, Double)
     - |uncheck|
   * - vector_function
     - The function to use for an mfem::VectorFunctionCoefficient
     - Vector(Vector, Double)
     - |uncheck|

---------------
vector_constant
---------------

Description: The constant vector to use as the coefficient

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - z
     - z-component of vector
     - 
     - 
     - |uncheck|
   * - y
     - y-component of vector
     - 
     - 
     - |uncheck|
   * - x
     - x-component of vector
     - 
     - 
     - |uncheck|

--------------
boundary_conds
--------------


--------------------
Collection contents:
--------------------

The input schema defines a collection of this container.
For brevity, only one instance is displayed here.

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - constant
     - The constant scalar value to use as the coefficient
     - 
     - 
     - |uncheck|
   * - component
     - The vector component to which the scalar coefficient should be applied
     - 
     - 
     - |uncheck|
.. list-table:: Functions
   :widths: 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Function Name
     - Description
     - Signature
     - Required
   * - scalar_function
     - The function to use for an mfem::FunctionCoefficient
     - Double(Vector, Double)
     - |uncheck|
   * - vector_function
     - The function to use for an mfem::VectorFunctionCoefficient
     - Vector(Vector, Double)
     - |uncheck|

---------------
vector_constant
---------------

Description: The constant vector to use as the coefficient

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - z
     - z-component of vector
     - 
     - 
     - |uncheck|
   * - y
     - y-component of vector
     - 
     - 
     - |uncheck|
   * - x
     - x-component of vector
     - 
     - 
     - |uncheck|

-----
attrs
-----


--------------------
Collection contents:
--------------------

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

------
source
------

Description: Scalar source term (RHS of the thermal conduction PDE)

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - constant
     - The constant scalar value to use as the coefficient
     - 
     - 
     - |uncheck|
   * - component
     - The vector component to which the scalar coefficient should be applied
     - 
     - 
     - |uncheck|
.. list-table:: Functions
   :widths: 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Function Name
     - Description
     - Signature
     - Required
   * - scalar_function
     - The function to use for an mfem::FunctionCoefficient
     - Double(Vector, Double)
     - |uncheck|
   * - vector_function
     - The function to use for an mfem::VectorFunctionCoefficient
     - Vector(Vector, Double)
     - |uncheck|

---------------
vector_constant
---------------

Description: The constant vector to use as the coefficient

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - z
     - z-component of vector
     - 
     - 
     - |uncheck|
   * - y
     - y-component of vector
     - 
     - 
     - |uncheck|
   * - x
     - x-component of vector
     - 
     - 
     - |uncheck|

------------------
nonlinear_reaction
------------------

Description: Nonlinear reaction term parameters

.. list-table:: Functions
   :widths: 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Function Name
     - Description
     - Signature
     - Required
   * - d_reaction_function
     - Derivative of the nonlinear reaction function dq = dq / dTemperature
     - Double(Double)
     - |uncheck|
   * - reaction_function
     - Nonlinear reaction function q = q(temperature)
     - Double(Double)
     - |uncheck|

-----
scale
-----

Description: Spatially varying scale factor for the reaction

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - constant
     - The constant scalar value to use as the coefficient
     - 
     - 
     - |uncheck|
   * - component
     - The vector component to which the scalar coefficient should be applied
     - 
     - 
     - |uncheck|
.. list-table:: Functions
   :widths: 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Function Name
     - Description
     - Signature
     - Required
   * - scalar_function
     - The function to use for an mfem::FunctionCoefficient
     - Double(Vector, Double)
     - |uncheck|
   * - vector_function
     - The function to use for an mfem::VectorFunctionCoefficient
     - Vector(Vector, Double)
     - |uncheck|

---------------
vector_constant
---------------

Description: The constant vector to use as the coefficient

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - z
     - z-component of vector
     - 
     - 
     - |uncheck|
   * - y
     - y-component of vector
     - 
     - 
     - |uncheck|
   * - x
     - x-component of vector
     - 
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
   * - enforcement_method
     - Time-varying constraint enforcement method to use
     - 
     - 
     - |uncheck|
   * - timestepper
     - Timestepper (ODE) method to use
     - 
     - 
     - |uncheck|

---------------
equation_solver
---------------

Description: Linear and Nonlinear stiffness Solver Parameters.


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
   * - solver_type
     - Solver type (MFEMNewton|KINFullStep|KINLineSearch)
     - MFEMNewton
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
   * - prec_type
     - Preconditioner type (JacobiSmoother|L1JacobiSmoother|AMG|BlockILU).
     - JacobiSmoother
     - 
     - |uncheck|
   * - solver_type
     - Solver type (gmres|minres|cg).
     - gmres
     - 
     - |uncheck|
   * - rel_tol
     - Relative tolerance for the linear solve.
     - 0.000001
     - 
     - |uncheck|
   * - max_iter
     - Maximum iterations for the linear solve.
     - 5000
     - 
     - |uncheck|
   * - abs_tol
     - Absolute tolerance for the linear solve.
     - 0.000000
     - 
     - |uncheck|
   * - print_level
     - Linear print level.
     - 0
     - 
     - |uncheck|

-----
solid
-----

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
   * - density
     - Initial mass density
     - 1.000000
     - 
     - |uncheck|
   * - viscosity
     - Viscosity constant
     - 0.000000
     - 
     - |uncheck|
   * - material_nonlin
     - Flag to include material nonlinearities (linear elastic vs. neo-Hookean material model).
     - True
     - 
     - |uncheck|
   * - order
     - Order degree of the finite elements.
     - 1
     - 1 to 8
     - |uncheck|
   * - K
     - Bulk modulus in the Neo-Hookean hyperelastic model.
     - 5.000000
     - 
     - |uncheck|
   * - mu
     - Shear modulus in the Neo-Hookean hyperelastic model.
     - 0.250000
     - 
     - |uncheck|
   * - geometric_nonlin
     - Flag to include geometric nonlinearities in the residual calculation.
     - True
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
   * - constant
     - The constant scalar value to use as the coefficient
     - 
     - 
     - |uncheck|
   * - component
     - The vector component to which the scalar coefficient should be applied
     - 
     - 
     - |uncheck|
.. list-table:: Functions
   :widths: 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Function Name
     - Description
     - Signature
     - Required
   * - scalar_function
     - The function to use for an mfem::FunctionCoefficient
     - Double(Vector, Double)
     - |uncheck|
   * - vector_function
     - The function to use for an mfem::VectorFunctionCoefficient
     - Vector(Vector, Double)
     - |uncheck|

---------------
vector_constant
---------------

Description: The constant vector to use as the coefficient

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - z
     - z-component of vector
     - 
     - 
     - |uncheck|
   * - y
     - y-component of vector
     - 
     - 
     - |uncheck|
   * - x
     - x-component of vector
     - 
     - 
     - |uncheck|

---------------
equation_solver
---------------

Description: Linear and Nonlinear stiffness Solver Parameters.


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
   * - solver_type
     - Solver type (MFEMNewton|KINFullStep|KINLineSearch)
     - MFEMNewton
     - 
     - |uncheck|
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
   * - prec_type
     - Preconditioner type (JacobiSmoother|L1JacobiSmoother|AMG|BlockILU).
     - JacobiSmoother
     - 
     - |uncheck|
   * - solver_type
     - Solver type (gmres|minres|cg).
     - gmres
     - 
     - |uncheck|
   * - rel_tol
     - Relative tolerance for the linear solve.
     - 0.000001
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
   * - abs_tol
     - Absolute tolerance for the linear solve.
     - 0.000000
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
   * - enforcement_method
     - Time-varying constraint enforcement method to use
     - 
     - 
     - |uncheck|
   * - timestepper
     - Timestepper (ODE) method to use
     - 
     - 
     - |uncheck|

--------------
boundary_conds
--------------


--------------------
Collection contents:
--------------------

The input schema defines a collection of this container.
For brevity, only one instance is displayed here.

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - constant
     - The constant scalar value to use as the coefficient
     - 
     - 
     - |uncheck|
   * - component
     - The vector component to which the scalar coefficient should be applied
     - 
     - 
     - |uncheck|
.. list-table:: Functions
   :widths: 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Function Name
     - Description
     - Signature
     - Required
   * - scalar_function
     - The function to use for an mfem::FunctionCoefficient
     - Double(Vector, Double)
     - |uncheck|
   * - vector_function
     - The function to use for an mfem::VectorFunctionCoefficient
     - Vector(Vector, Double)
     - |uncheck|

---------------
vector_constant
---------------

Description: The constant vector to use as the coefficient

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - y
     - y-component of vector
     - 
     - 
     - |uncheck|
   * - z
     - z-component of vector
     - 
     - 
     - |uncheck|
   * - x
     - x-component of vector
     - 
     - 
     - |uncheck|

-----
attrs
-----


--------------------
Collection contents:
--------------------

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
   * - constant
     - The constant scalar value to use as the coefficient
     - 
     - 
     - |uncheck|
   * - component
     - The vector component to which the scalar coefficient should be applied
     - 
     - 
     - |uncheck|
.. list-table:: Functions
   :widths: 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Function Name
     - Description
     - Signature
     - Required
   * - scalar_function
     - The function to use for an mfem::FunctionCoefficient
     - Double(Vector, Double)
     - |uncheck|
   * - vector_function
     - The function to use for an mfem::VectorFunctionCoefficient
     - Vector(Vector, Double)
     - |uncheck|

---------------
vector_constant
---------------

Description: The constant vector to use as the coefficient

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - z
     - z-component of vector
     - 
     - 
     - |uncheck|
   * - y
     - y-component of vector
     - 
     - 
     - |uncheck|
   * - x
     - x-component of vector
     - 
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
   * - approx_elements
     - Approximate number of elements in an n-ball mesh
     - 
     - 
     - |uncheck|
   * - ser_ref_levels
     - Number of times to refine the mesh uniformly in serial.
     - 0
     - 
     - |uncheck|
   * - mesh
     - Path to Mesh file
     - 
     - 
     - |uncheck|
   * - par_ref_levels
     - Number of times to refine the mesh uniformly in parallel.
     - 0
     - 
     - |uncheck|
   * - type
     - Type of mesh
     - 
     - ball, box, disk, file
     - |check|

----
size
----

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - y
     - Size in the y-dimension
     - 
     - 
     - |uncheck|
   * - z
     - Size in the z-dimension
     - 
     - 
     - |uncheck|
   * - x
     - Size in the x-dimension
     - 
     - 
     - |uncheck|

--------
elements
--------

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - z
     - z-dimension
     - 
     - 
     - |uncheck|
   * - y
     - y-dimension
     - 
     - 
     - |uncheck|
   * - x
     - x-dimension
     - 
     - 
     - |uncheck|
