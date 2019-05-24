#ifndef LOADING_FUNCTIONS
#define LOADING_FUNCTIONS

#include "mfem.hpp"

using namespace mfem;

// set kinematic functions and boundary condition functions
void ReferenceConfiguration(const Vector &x, Vector &y);
void InitialDeformation(const Vector &x, Vector &y);

#endif
