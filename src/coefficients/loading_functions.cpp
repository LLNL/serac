#include "loading_functions.hpp"

using namespace mfem;

void ReferenceConfiguration(const Vector &x, Vector &y)
{
   // set the reference, stress free, configuration
   y = x;
}


void InitialDeformation(const Vector &x, Vector &y)
{
   y = x;
   y = 0.0;
}


