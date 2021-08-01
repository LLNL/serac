#include <vector>
#include <random>

#include "axom/core/utilities/Timer.hpp"

#include "serac/physics/utilities/functional/tensor.hpp"

using namespace serac;

template <typename lambda>
auto time(lambda&& f)
{
  axom::utilities::Timer stopwatch;
  stopwatch.start();
  f();
  stopwatch.stop();
  return stopwatch.elapsed();
}

auto random_real = [](auto...) {
  static std::default_random_engine             generator;
  static std::uniform_real_distribution<double> distribution(0.0, 1.0);
  return distribution(generator);
};

template < int m, int n >
struct BSRMatrix {
  std::vector < int > row_ptr; 
  std::vector < int > col_ind; 
  std::vector < tensor< double, m, n > > values;

  std::vector < double > output;

  static BSRMatrix<m,n> random(int num_block_rows, int num_block_cols, double sparsity = 0.1) {
    BSRMatrix<m,n> A{};

    double scale = 2.0 * ((1.0 / sparsity) - 1.0);
    double t = scale * random_real();
    int nnz = 0;
    int previous_row = 0;
    A.output.resize(num_block_rows * m);
    A.row_ptr.resize(num_block_rows+1);
    A.col_ind.reserve(int(1.1 * num_block_rows * num_block_cols * sparsity));
    A.values.reserve(int(1.1 * num_block_rows * num_block_cols * sparsity));
    A.row_ptr[0] = 0;
    do {
      nnz++;
      int block_row = int(t) / num_block_cols;
      int block_col = int(t) % num_block_cols;
      for (int r = previous_row; r < block_row; r++) { A.row_ptr[r+1] = nnz - 1; }
      A.col_ind.push_back(block_col);
      A.values.push_back(make_tensor<m,n>(random_real));
      previous_row = block_row;
      t += 1 + scale * random_real();
    } while(t < num_block_rows * num_block_cols);
    A.row_ptr.back() = nnz;
    return A;
  }

};

struct CSRMatrix {
  std::vector < int > row_ptr; 
  std::vector < int > col_ind; 
  std::vector < double > values;

  std::vector < double > output;

  template < int m, int n >
  static CSRMatrix from_BSRMatrix(const BSRMatrix< m, n > & A) {
    CSRMatrix A_csr{};
    size_t block_rows = A.row_ptr.size() - 1;
    size_t block_nnz = A.col_ind.size();
    A_csr.output.resize(block_rows * m + 1);
    A_csr.row_ptr.resize(block_rows * m + 1);
    A_csr.col_ind.resize(block_nnz * m * n);
    A_csr.values.resize(block_nnz * m * n);

    for (size_t r = 0; r < A.row_ptr.size() - 1; r++) {
      int base_id = A.row_ptr[r] * m * n;
      int offset_per_row = n * (A.row_ptr[r+1] - A.row_ptr[r]);
      for (int i = 0; i < m; i++) {
        A_csr.row_ptr[m * r + i] = base_id + i * offset_per_row;
      }

      for (int k = A.row_ptr[r]; k < A.row_ptr[r+1]; k++) {
        int c = A.col_ind[k];
        for (int i = 0; i < m; i++) {
          for (int j = 0; j < n; j++) {
            int id = base_id + i * offset_per_row + n * (k - A.row_ptr[r])  + j;
            A_csr.col_ind[id] = n * c + j;
            A_csr.values[id] = A.values[k][i][j];
          }
        }
      }
    }

    A_csr.row_ptr.back() = A.row_ptr.back() * m * n;

    return A_csr;

  }

};

template < int m, int n >
std::vector < double > & operator*(BSRMatrix<m,n> & A, const std::vector<double> x) {
  auto blocked_x = reinterpret_cast< const tensor< double, n > * >(x.data());
  auto blocked_b = reinterpret_cast< tensor< double, m > * >(A.output.data());

  for (size_t r = 0; r < A.row_ptr.size() - 1; r++) {
    blocked_b[r] = tensor<double, m>{};
    for (int k = A.row_ptr[r]; k < A.row_ptr[r+1]; k++) {
      int c = A.col_ind[k];
      blocked_b[r] += dot(A.values[k], blocked_x[c]);
    }
  }
  return A.output;
}

std::vector < double > & operator*(CSRMatrix & A, const std::vector<double> x) {
  for (size_t r = 0; r < A.row_ptr.size() - 1; r++) {
    A.output[r] = 0.0;
    for (int k = A.row_ptr[r]; k < A.row_ptr[r+1]; k++) {
      int c = A.col_ind[k];
      A.output[r] += A.values[k] * x[c];
    }
  }
  return A.output;
}

int main() {

  constexpr int m = 3;
  constexpr int n = 3;
  double sparsity = 0.03;
  int num_block_rows = 1000;
  int num_block_cols = 1000;

  BSRMatrix A1 = BSRMatrix<m, n>::random(num_block_rows, num_block_cols, sparsity);
  CSRMatrix A2 = CSRMatrix::from_BSRMatrix(A1);

  std::vector < double > x(num_block_cols * n, 1.0);

  auto b1 = A1 * x;
  auto b2 = A2 * x;

  for (size_t i = 0; i < b1.size(); i++) {      
    std::cout << i << " " << b1[i] << " " << b2[i] << std::endl;
  }

  std::cout << "BSR time: " << time([&](){ for (int i = 0; i < 10; i++) { A1 * x; } }) << std::endl;
  std::cout << "CSR time: " << time([&](){ for (int i = 0; i < 10; i++) { A2 * x; } }) << std::endl;

}
