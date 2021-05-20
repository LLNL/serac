.. ## Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
.. ## other Serac Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

.. _header-n2:

=========
Weak Form
=========

``WeakForm`` is a class that is used specify and evaluate
finite-element-type calculations. For example, the weighted residual for
a solid mechanics simulation may look something like:

.. math::

   r(u) := 
   \underbrace{\int_\Omega \nabla\psi : \sigma(\nabla u) \; \text{d}v}_{\text{stress response}}
   \;+\;
   \underbrace{\int_\Omega \psi \cdot b(\mathbf{x}) \; \text{d}v}_{\text{body forces}} 
   \;+\;
   \underbrace{\int_{\partial\Omega} \psi \cdot \mathbf{t}(\mathbf{x}) \; \text{d}a}_{\text{surface loads}},

where :math:`\psi` are the test basis functions. To describe this
residual using ``WeakForm``, we first create the object itself, providing a
template parameter that expresses the test and trial spaces (i.e. the
"inputs" and "outputs" of the residual function, :math:`r`). In this
case, solid mechanics uses nodal displacements and residuals (i.e. H1 test and trial spaces), so we write:

.. code-block:: cpp

   constexpr int order = 2; // the polynomial order of the basis functions
   constexpr int dim = 3; // the dimension of our problem
   using test = H1<order, dim>;
   using trial = H1<order, dim>;
   WeakForm< test(trial) > residual(test_fes, trial_fes);

where ``test_fes``, ``trial_fes`` are the ``mfem`` finite element spaces for the problem. 
The template argument follows the same convention of ``std::function``:
the output-type appears outside the parentheses, and the input-type(s)
appear, inside the parentheses (in order). So, the last line of code in
the snippet above is saying that ``residual`` is going to represent a
calculation that takes in an H1 field (displacements), and returns a
vector of weighted residuals, using H1 test functions.

Now that the ``WeakForm`` object is created, we can use the
following functions to define integral terms (depending on their
dimensionality). Here, we use :math:`s` to denote the "source" term
(integrated against test functions), and :math:`f` to denote the 
"flux" term (integrated against test function gradients).

1. Integrals of the form:
   :math:`\displaystyle \iint_\Omega \psi \cdot s + \nabla \psi : f \; da`

   .. code-block:: cpp

      WeakForm::AddAreaIntegral([](auto ... args){
      	auto s = ...;
      	auto f = ...;
      	return std::tuple{s, f};
      }, domain_of_integration);

2. Integrals of the form:
   :math:`\displaystyle \iiint_\Omega \psi \cdot s + \nabla \psi : f \; dv`

   .. code-block:: cpp

      WeakForm::AddVolumeIntegral([](auto ... args){
      	auto s = ...;
      	auto f = ...;
      	return std::tuple{s, f};
      }, domain_of_integration);

3. Integrals of the form:
   :math:`\displaystyle \iint_{\partial \Omega} \psi \cdot s \; da`

   .. code-block:: cpp

      WeakForm::AddSurfaceIntegral([](auto ... args){
      	auto s = ...;
      	return s;
      }, domain_of_integration);	

So, for our problem (since we assumed 3D earlier) we can make an
``Add****Integral()`` call for each of the integral terms in the
original residual. In each of these functions, the first argument is the
integrand (a lambda function or functor returning :math:`\{s, f\}`),
and the second argument is the domain of integration. Let's start with
the stress response term:

.. code-block:: cpp

   // The integrand lambda function is passed the spatial position of the quadrature point,
   // as well as a {value, derivative} tuple for the trial space.
   residual.AddVolumeIntegral([](auto x, auto disp){
     
     // Here, we unpack the {value, derivative} tuple into separate variables
     auto [u, grad_u] = disp;
     
     // WeakForm expects us to return a tuple of the form {s, f} (see table above)
     auto body_force = zero{}; // for this case, the source term is identically zero.
     auto stress = material_model(grad_u); // call some constitutive model for the material in this domain
     
     return std::tuple{body_force, stress};
     
   }, mesh);

The other terms follow a similar pattern. For the body force:

.. code-block:: cpp

   residual.AddVolumeIntegral([](auto x, auto disp /* unused */){
     
     // WeakForm::AddVolumeIntegral() expects us to return a tuple of the form {s, f}
     auto body_force = b(x); // evaluate the body-force at the location of the quadrature point
     auto stress = zero{}; // for this term, the stress term is identically zero
     
     return std::tuple{body_force, stress};
     
   }, mesh);

And finally, for the surface tractions:

.. code-block:: cpp

   // WeakForm::AddSurfaceIntegral() only expects us to return s, so we don't need a tuple
   residual.AddSurfaceIntegral([](auto x, auto disp /* unused */){
     return traction(x); // evaluate the traction at the location of the quadrature point
   }, surface_mesh);

Now that we've finished describing all the integral terms that appear in
our residual, we can carry out the actual calculation by calling
``WeakForm::operator()``:

.. code-block:: cpp

   auto r = residual(displacements);

Putting these snippets together without the verbose comments, we have (note: the two AddVolumeIntegrals were fused into one):

.. code-block:: cpp

   using test = H1<order, dim>;
   using trial = H1<order, dim>;
   WeakForm< test(trial) > residual(test_fes, trial_fes);

   // note: the first two AddVolumeIntegral calls can be fused
   // into one, provided they share the same domain of integration
   residual.AddVolumeIntegral([](auto x, auto disp){
     auto [u, grad_u] = disp;
     return std::tuple{b(x), material_model(grad_u))};
   }, mesh);

   residual.AddSurfaceIntegral([](auto x, auto disp /* unused */){ return traction(x); }, surface_mesh);

   auto r = residual(displacements);

So, in only a few lines of code, we can create optimized, custom finite
element kernels!

.. _header-n34:

Implementation
--------------

For the most part, the ``WeakForm`` class is just a container of
``Integral`` objects, and some prolongation and restriction operators to
get the data they need:

.. code-block:: cpp

   template <typename test, typename trial>
   struct WeakForm<test(trial)> : public mfem::Operator {
     ...
     std::vector< Integral<test(trial)> > domain_integrals;
     std::vector< Integral<test(trial)> > boundary_integrals;
   };

The calls to ``WeakForm::Add****Integral`` forward the integrand and
mesh information to an ``Integral`` constructor and add it to the
appropriate list (either ``domain_integrals`` or
``boundary_integrals``). MFEM treats domain and boundary integrals
differently, so we maintain them in separate lists.

From there, the ``Integral`` constructor uses the integrand functor to
specialize a highly templated finite element kernel (simplified
implementation given below).

.. code-block:: cpp

   template < ::Geometry g, typename test, typename trial, int geometry_dim, int spatial_dim, int Q,
              typename derivatives_type, typename lambda>
   void evaluation_kernel(const mfem::Vector& U, mfem::Vector& R, derivatives_type* derivatives_ptr,
                          const mfem::Vector& J_, const mfem::Vector& X_, int num_elements, lambda qf)
   {
     ...

     // for each element in the domain
     for (int e = 0; e < num_elements; e++) {
     
       // get the values for this particular element
       tensor u_elem = detail::Load<trial_element>(u, e);

       // this is where we will accumulate the element residual tensor
       element_residual_type r_elem{};

       // for each quadrature point in the element
       for (int q = 0; q < static_cast<int>(rule.size()); q++) {
         // get the position of this quadrature point in the parent and physical space,
         // and calculate the measure of that point in physical space.
         auto   xi  = rule.points[q];
         auto   dxi = rule.weights[q];
         auto   x_q = make_tensor<spatial_dim>([&](int i) { return X(q, i, e); });
         auto   J_q = make_tensor<spatial_dim, geometry_dim>([&](int i, int j) { return J(q, i, j, e); });
         double dx  = detail::Measure(J_q) * dxi;

         // evaluate the value/derivatives needed for the q-function at this quadrature point
         auto arg = detail::Preprocess<trial_element>(u_elem, xi, J_q);

         // evaluate the user-specified constitutive model
         //
         // note: make_dual(arg) promotes those arguments to dual number types
         // so that qf_output will contain values and derivatives
         auto qf_output = qf(x_q, make_dual(arg));

         // integrate qf_output against test space shape functions / gradients
         // to get element residual contributions
         r_elem += detail::Postprocess<test_element>(get_value(qf_output), xi, J_q) * dx;
         
       }

       // once we've finished the element integration loop, write our element residuals
       // out to memory, to be later assembled into global residuals by mfem
       detail::Add(r, r_elem, e);
     }
   }

Then, the call to that specialized finite element kernel is wrapped
inside a ``std::function`` object with the appropriate signature. This
``std::function`` is used to implement the action of ``Mult()``:

.. code-block:: cpp

   template < typename spaces > 
   struct Integral {

     ...
     
     template <int geometry_dim, int spatial_dim, typename lambda_type>
     Integral(...) {

       ...
       
       evaluation = [=](const mfem::Vector& U, mfem::Vector& R) {
         evaluation_kernel<geometry, test_space, trial_space, geometry_dim, spatial_dim, Q>(...);
       };
       
       ...
       
     };
     
     void Mult(const mfem::Vector& input, mfem::Vector& output) const { evaluation(input, output); }
     
     std::function<void(const mfem::Vector&, mfem::Vector&)> evaluation;
     
   }

Finally, when the user calls ``WeakForm::operator()``, it loops over the
domain and surface integrals, calling ``Integral::Mult()`` on each one
to compute the weighted residual contribution from each term.
