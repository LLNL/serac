// minimal reproducer for https://github.com/mfem/mfem/issues/3641
// expected to be deleted once issue is resolved
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

double abs_tolerance = 1.0e-12;
double rel_tolerance = 1.0e-6;

struct SimpleFunction : mfem::Operator {

    SimpleFunction() : mfem::Operator(3, 3), A(3, 3), b(3) {
        A(0,0) = 3.0; A(0,1) = 0.0; A(0,2) = 0.0; b(0) = 1;
        A(1,0) = 0.0; A(1,1) = 2.0; A(1,2) = 0.0; b(1) = 2;
        A(2,0) = 0.0; A(2,1) = 0.0; A(2,2) = 1.0; b(2) = 3;
    }

    void Mult(const mfem::Vector & x, mfem::Vector & r) const {
        // r := A * x - b
        A.Mult(x, r);
        r -= b;
    }

    mfem::Operator & GetGradient(const mfem::Vector & x) const {
        return A;
    }

    mutable mfem::DenseMatrix A;
    mfem::Vector b;

};


struct SimplePreconditioner : mfem::Solver {

    SimplePreconditioner() : mfem::Solver(3, 3), M(3, 3) {
        M(0,0) = 1.0 / 3.0; 
        M(1,1) = 1.0 / 2.0; 
        M(2,2) = 1.0 + abs_tolerance;
    }

    void SetOperator(const mfem::Operator & op) {
        // don't care
    }

    void Mult(const mfem::Vector & x, mfem::Vector & r) const {
        M.Mult(x, r);
    }

    mfem::DenseMatrix M;

};

int main() {

    SimpleFunction f;
    SimplePreconditioner M;
    mfem::GMRESSolver krylov;
    krylov.SetPreconditioner(M);
    krylov.SetRelTol(rel_tolerance);
    krylov.SetAbsTol(abs_tolerance);
    krylov.iterative_mode = false;
    krylov.SetPrintLevel(1);

    mfem::NewtonSolver finv; 
    finv.SetOperator(f);
    finv.SetSolver(krylov);
    finv.SetRelTol(rel_tolerance);
    finv.SetAbsTol(abs_tolerance);
    finv.iterative_mode = true;
    finv.SetPrintLevel(1);

    mfem::Vector x(3);
    x[0] = (1.0 / 3.0) + abs_tolerance;
    x[1] = 1.0;
    x[2] = 3.0;

    mfem::Vector zero(3);
    zero[0] = 0;
    zero[1] = 0;
    zero[2] = 0;

    finv.Mult(zero, x);

    x.Print(std::cout);

}
