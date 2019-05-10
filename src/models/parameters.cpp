#include "parameters.hpp"

FlobatParams::FlobatParams()
{
   m_params.SetSize(FlobatParams::numParams);
   m_params = -1.0;
}

FlobatParams::FlobatParams(const FlobatParams &p2)
{
   m_params = p2.m_params;
}

void FlobatParams::SetValue(int param, double val)
{
   m_params[param] = val;
}

double FlobatParams::GetValue(int param)
{
   return m_params[param];
}
