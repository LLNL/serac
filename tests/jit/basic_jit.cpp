#include <iostream>

#include "JIT.hpp"

int main() {

  JIT jit({/* compilation flags */});

  std::cout << "parsing function template" << std::endl;
  jit.compile(R"(
    template < int m > 
    double foo() {
      return m + 2;
    }
  )");

  std::cout << "instantiating function template specialization" << std::endl;
  jit.compile(R"(
    extern "C" {
      double foo_specialization2() {
        return foo<2>();
      }
    }
  )");

  std::cout << "get function pointer to specialization" << std::endl;
  double (*foo2)() = jit.lookup_function<double(*)()>("foo_specialization2");

  std::cout << "calling foo<2>()" << std::endl;
  std::cout << foo2() << std::endl;
 
  return 0;
}