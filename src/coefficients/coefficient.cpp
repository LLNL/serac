#include "coefficients/coefficient.hpp"

using namespace serac;

coefficient::coefficient(double value) : self_(std::make_unique<mfem::ConstantCoefficient>(value)) {}
coefficient::coefficient(std::function <double(const mfem::Vector &)> f) : self_(std::make_unique<StdFunctionCoefficient>(f)) {}
coefficient::coefficient(FiniteElementState & fes) : self_(std::make_unique<mfem::GridFunctionCoefficient>(&fes.gridFunc())) {}

vector_coefficient::vector_coefficient(mfem::Vector value) : self_(std::make_unique<mfem::VectorConstantCoefficient>(value)) {}
vector_coefficient::vector_coefficient(int components, std::function <void(const mfem::Vector &, mfem::Vector &)> f) : self_(std::make_unique<StdFunctionVectorCoefficient>(components, f)) {}