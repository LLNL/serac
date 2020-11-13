#include "coefficients/whatever_coefficient.hpp"

using namespace serac;

whatever_coefficient::whatever_coefficient(double value) : self_(std::make_unique<mfem::ConstantCoefficient>(value)) {}
whatever_coefficient::whatever_coefficient(std::function <double(const mfem::Vector &)> f) : self_(std::make_unique<StdFunctionCoefficient>(f)) {}

whatever_vector_coefficient::whatever_vector_coefficient(mfem::Vector value) : self_(std::make_unique<mfem::VectorConstantCoefficient>(value)) {}
whatever_vector_coefficient::whatever_vector_coefficient(int components, std::function <void(const mfem::Vector &, mfem::Vector &)> f) : self_(std::make_unique<StdFunctionVectorCoefficient>(components, f)) {}