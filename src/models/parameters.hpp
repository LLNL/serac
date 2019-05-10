#ifndef FLOBAT_PARAMS
#define FLOBAT_PARAMS

#include "mfem.hpp"

using namespace mfem;

class FlobatParams
{
public:
   enum Parameters
   {
      epsilon0 = 0,
      DO0,
      DR0,      
      kappa0,
      sigma0,
      a0,
      Cref,
      U0,
      i0,
      kmR,
      kmO,
      nF,
      alphaAFRT,
      alphaCFRT,
      I,
      A,
      numParams
   };

   FlobatParams();
   FlobatParams(const FlobatParams &p2);
   ~FlobatParams();

   void SetValue(int param, double val);
   double GetValue(int param);
                
private:
   Array<double> m_params;
   
};

#endif
