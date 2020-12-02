#include "coefficients/coefficient.hpp"

using namespace serac;

CoefficientWrapper::CoefficientWrapper(double value) : self_(std::make_unique<mfem::ConstantCoefficient>(value)) {}
CoefficientWrapper::CoefficientWrapper(std::function<double(const mfem::Vector&)> f)
    : self_(std::make_unique<mfem::FunctionCoefficient>(f))
{
}
CoefficientWrapper::CoefficientWrapper(std::function<double(const mfem::Vector&, double)> f)
    : self_(std::make_unique<mfem::FunctionCoefficient>(f))
{
}
CoefficientWrapper::CoefficientWrapper(const FiniteElementState& fes)
    : self_(std::make_unique<mfem::GridFunctionCoefficient>(&fes.gridFunc()))
{
}
CoefficientWrapper::CoefficientWrapper(CoefficientWrapper&& coef, std::function<double(const double)>&& f)
    : self_(std::make_unique<TransformedCoefficient>(std::move(coef), std::move(f)))
{
}

VectorCoefficientWrapper::VectorCoefficientWrapper(mfem::Vector value)
    : self_(std::make_unique<mfem::VectorConstantCoefficient>(value))
{
}
VectorCoefficientWrapper::VectorCoefficientWrapper(int                                                     components,
                                                   std::function<void(const mfem::Vector&, mfem::Vector&)> f)
    : self_(std::make_unique<mfem::VectorFunctionCoefficient>(components, f))
{
}