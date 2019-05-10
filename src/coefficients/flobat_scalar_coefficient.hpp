#ifndef FLOBAT_SCALAR_COEFFICIENT
#define FLOBAT_SCALAR_COEFFICIENT

#include "mfem.hpp"
#include "models/parameters.hpp"

using namespace mfem;

class FlobatScalarCoefficient : public Coefficient
{
public:
   FlobatScalarCoefficient(Coefficient & gamma, Coefficient & phi1, Coefficient & phi2, Coefficient & CO,
                           Coefficient & CR, VectorCoefficient & vel, FlobatParams & params);

   void SetFields(Coefficient & gamma, Coefficient & phi1, Coefficient & phi2, Coefficient & CO,
                  Coefficient & CR, VectorCoefficient & vel);

   void SetParameters(FlobatParams & params);
   
   /// Evaluate the coefficient
   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip) = 0;
protected:
   
   Coefficient *m_gamma;
   Coefficient *m_phi1;
   Coefficient *m_phi2;
   Coefficient *m_CR;
   Coefficient *m_CO;
   VectorCoefficient *m_vel;
   
   FlobatParams m_params;   
};

class PotentialSourceCoefficient : public FlobatScalarCoefficient
{
public:
   PotentialSourceCoefficient(Coefficient & gamma, Coefficient & phi1, Coefficient & phi2, Coefficient & CO,
                         Coefficient & CR, VectorCoefficient & vel, FlobatParams & params) :
      FlobatScalarCoefficient(gamma, phi1, phi2, CO, CR, vel, params) {};

   double Eval(ElementTransformation &T, const IntegrationPoint &ip);
   
};

class ConvectionSourceCoefficient : public FlobatScalarCoefficient
{
public:
   ConvectionSourceCoefficient(Coefficient & gamma, Coefficient & phi1, Coefficient & phi2, Coefficient & CO,
                         Coefficient & CR, VectorCoefficient & vel, FlobatParams & params) :
      FlobatScalarCoefficient(gamma, phi1, phi2, CO, CR, vel, params) {};

   double Eval(ElementTransformation &T, const IntegrationPoint &ip);
   
};

class OxidantDiffusionCoefficient : public FlobatScalarCoefficient
{
public:
   OxidantDiffusionCoefficient(Coefficient & gamma, Coefficient & phi1, Coefficient & phi2, Coefficient & CO,
                          Coefficient & CR, VectorCoefficient & vel, FlobatParams & params) :
      FlobatScalarCoefficient(gamma, phi1, phi2, CO, CR, vel, params) {};

   double Eval(ElementTransformation &T, const IntegrationPoint &ip);
   
};

class ReductantDiffusionCoefficient : public FlobatScalarCoefficient
{
public:
   ReductantDiffusionCoefficient(Coefficient & gamma, Coefficient & phi1, Coefficient & phi2, Coefficient & CO,
                                 Coefficient & CR, VectorCoefficient & vel, FlobatParams & params) :
      FlobatScalarCoefficient(gamma, phi1, phi2, CO, CR, vel, params) {};
   
   double Eval(ElementTransformation &T, const IntegrationPoint &ip);
   
};

class KappaCoefficient : public FlobatScalarCoefficient
{
public:
   KappaCoefficient(Coefficient & gamma, Coefficient & phi1, Coefficient & phi2, Coefficient & CO,
                    Coefficient & CR, VectorCoefficient & vel, FlobatParams & params) :
      FlobatScalarCoefficient(gamma, phi1, phi2, CO, CR, vel, params) {};
   
   double Eval(ElementTransformation &T, const IntegrationPoint &ip);
   
};

class SigmaCoefficient : public FlobatScalarCoefficient
{
public:
   SigmaCoefficient(Coefficient & gamma, Coefficient & phi1, Coefficient & phi2, Coefficient & CO,
                    Coefficient & CR, VectorCoefficient & vel, FlobatParams & params) :
      FlobatScalarCoefficient(gamma, phi1, phi2, CO, CR, vel, params) {};
   
   double Eval(ElementTransformation &T, const IntegrationPoint &ip);
   
};

#endif
