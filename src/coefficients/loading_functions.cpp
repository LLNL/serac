#include "loading_functions.hpp"

using namespace mfem;

void ReferenceConfiguration(const Vector &x, Vector &y)
{
   // set the reference, stress free, configuration
   y = x;
}


void InitialDeformation(__attribute__((unused)) const Vector &x, Vector &y)
{
   y = 0.0;
}


