#include "serac/physics/utilities/functional/array.hpp"

void my_kernel(axom::CPUView< double, 2 > values) {
  for (int i = 0; i < 32; i++) {
    for (int j = 0; j < 32; j++) {
      values(i, j) = i + j;
    }
  }
} 

int main() {

  axom::CPUArray< double, 2 > my_array(32, 32);
  axom::CPUArray< double, 3 > my_array2(33, 32, 4);

  my_kernel(view(my_array));

}
