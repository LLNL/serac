#include "flobat_scalar_coefficient.hpp"
#include "models/parameters.hpp"

using namespace mfem;

FlobatScalarCoefficient::FlobatScalarCoefficient(Coefficient & gamma, Coefficient & phi1, Coefficient & phi2, Coefficient & CO,
                                                 Coefficient & CR, VectorCoefficient & vel, FlobatParams & params) :
   m_gamma(&gamma), m_phi1(&phi1), m_phi2(&phi2), m_CR(&CR), m_CO(&CO), m_vel(&vel), m_params(params)
{
}

void FlobatScalarCoefficient::SetFields(Coefficient & gamma, Coefficient & phi1, Coefficient & phi2, Coefficient & CO,
                                        Coefficient & CR, VectorCoefficient & vel)
{

   m_gamma = &gamma;
   m_phi1 = &phi1;
   m_phi2 = &phi2;
   m_CO = &CO;
   m_CR = &CR;
   m_vel = &vel;
   
}

void FlobatScalarCoefficient::SetParameters(FlobatParams & params)
{
   m_params = params;
}

double PotentialSourceCoefficient::Eval(ElementTransformation &T, const IntegrationPoint &ip)
{

   double gamma = m_gamma->Eval(T, ip);
   double phi1 = m_phi1->Eval(T, ip);
   double phi2 = m_phi2->Eval(T, ip);      
   double CO = m_CO->Eval(T, ip);
   double CR = m_CR->Eval(T, ip);
   Vector vel;
   m_vel->Eval(vel, T, ip);

   double U0 = m_params.GetValue(FlobatParams::U0);
   double i0 = m_params.GetValue(FlobatParams::i0);
   double Cref = m_params.GetValue(FlobatParams::Cref);
   double alphaAFRT = m_params.GetValue(FlobatParams::alphaAFRT);
   double alphaCFRT = m_params.GetValue(FlobatParams::alphaCFRT);
   double nF = m_params.GetValue(FlobatParams::nF);
   double kmR = m_params.GetValue(FlobatParams::kmR);
   double kmO = m_params.GetValue(FlobatParams::kmO);               
   double a0 = m_params.GetValue(FlobatParams::a0);
   
   double delphi = phi1 - phi2 - U0;
   double in = (i0/Cref) *
      (CR * std::exp(alphaAFRT * delphi) - CO * std::exp(-1.0 * alphaCFRT * delphi)) /
      (1.0 + (i0/nF) *
       ((std::exp(alphaAFRT * delphi)/kmR) +
        (std::exp(-1.0 * alphaCFRT * delphi)/kmO)));

   in *= a0 * gamma;
   
   return in;
   
}

double ConvectionSourceCoefficient::Eval(ElementTransformation &T, const IntegrationPoint &ip)
{

   double gamma = m_gamma->Eval(T, ip);
   double phi1 = m_phi1->Eval(T, ip);
   double phi2 = m_phi2->Eval(T, ip);      
   double CO = m_CO->Eval(T, ip);
   double CR = m_CR->Eval(T, ip);
   Vector vel;
   m_vel->Eval(vel, T, ip);

   double U0 = m_params.GetValue(FlobatParams::U0);
   double i0 = m_params.GetValue(FlobatParams::i0);
   double Cref = m_params.GetValue(FlobatParams::Cref);
   double alphaAFRT = m_params.GetValue(FlobatParams::alphaAFRT);
   double alphaCFRT = m_params.GetValue(FlobatParams::alphaCFRT);
   double nF = m_params.GetValue(FlobatParams::nF);
   double kmR = m_params.GetValue(FlobatParams::kmR);
   double kmO = m_params.GetValue(FlobatParams::kmO);               
   double a0 = m_params.GetValue(FlobatParams::a0);
   
   double delphi = phi1 - phi2 - U0;
   double jn = (CR * std::exp(alphaAFRT * delphi) - CO * std::exp(-1.0 * alphaCFRT * delphi)) /
      ((nF*Cref)/i0 + 
       ((std::exp(alphaAFRT * delphi)/kmR) +
        (std::exp(-1.0 * alphaCFRT * delphi)/kmO)));
   
   jn *= a0 * gamma;
   
   return jn;
   
}

double OxidantDiffusionCoefficient::Eval(ElementTransformation &T, const IntegrationPoint &ip)
{

   double gamma = m_gamma->Eval(T, ip);

   double epsilon0 = m_params.GetValue(FlobatParams::epsilon0);
   double DO0 = m_params.GetValue(FlobatParams::DO0);   
   
   double DO = std::pow(1.0 - gamma * (1.0 - epsilon0), 1.5) * DO0;
   
   return DO;
   
}

double ReductantDiffusionCoefficient::Eval(ElementTransformation &T, const IntegrationPoint &ip)
{

   double gamma = m_gamma->Eval(T, ip);

   double epsilon0 = m_params.GetValue(FlobatParams::epsilon0);
   double DR0 = m_params.GetValue(FlobatParams::DR0);   
   
   double DR = std::pow(1.0 - gamma * (1.0 - epsilon0), 1.5) * DR0;
   
   return DR;
   
}

double KappaCoefficient::Eval(ElementTransformation &T, const IntegrationPoint &ip)
{

   double gamma = m_gamma->Eval(T, ip);

   double epsilon0 = m_params.GetValue(FlobatParams::epsilon0);
   double kappa0 = m_params.GetValue(FlobatParams::kappa0);   
   
   double kappa = std::pow(1.0 - gamma * (1.0 - epsilon0), 1.5) * kappa0;
   
   return kappa;
   
}

double SigmaCoefficient::Eval(ElementTransformation &T, const IntegrationPoint &ip)
{

   double gamma = m_gamma->Eval(T, ip);

   double epsilon0 = m_params.GetValue(FlobatParams::epsilon0);
   double sigma0 = m_params.GetValue(FlobatParams::sigma0);   
   
   double sigma = std::pow(gamma * (1.0 - epsilon0), 1.5) * sigma0;
   
   return sigma;
   
}
