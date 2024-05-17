#include <iostream>

#include "serac/numerics/functional/enzyme_declarations.hpp"

double square(double x) {
  return x * x;
}

double dsquare(double x) {
  return __enzyme_autodiff<double>(reinterpret_cast<void*>(square), x);
}

int main() {
  for(double i=1; i<5; i++) {
    std::cout << square(i) << " " << dsquare(i) << std::endl;
  }
}
