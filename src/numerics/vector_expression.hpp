// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file vector_expression.hpp
 *
 * @brief A set of template classes used to represent the evaluation of unary and binary operations
 * on vectors
 */

#ifndef VECTOR_EXPRESSION
#define VECTOR_EXPRESSION

#include "mfem.hpp"

namespace serac {
/**
 * @brief A base class representing a vector expression
 * @tparam T The base vector type, e.g., mfem::Vector, or another VectorExpr
 * @note This class should never be used directly
 */
template <typename T>
class VectorExpr {
public:
  /**
   * @brief Returns the fully evaluated value for the vector
   * expression at index @p i
   * @param i The index to evaluate at
   */
  double operator[](size_t i) const { return asDerived()[i]; }

  /**
   * @brief Returns the size of the vector expression
   */
  size_t Size() const { return asDerived().Size(); }

  /**
   * @brief Implicit conversion operator for fully evaluating
   * a vector expression into an actual mfem::Vector
   * @return The fully evaluated vector
   */
  operator mfem::Vector() const
  {
    mfem::Vector result(Size());
    for (size_t i = 0; i < Size(); i++) {
      result[i] = (*this)[i];
    }
    return result;
  }

  /**
   * @brief Performs a compile-time downcast to the derived object
   * @return The derived object
   * @see Curiously Recurring Template Pattern
   */
  const T& asDerived() const { return static_cast<const T&>(*this); }

  /**
   * @brief Performs a compile-time downcast to the derived object
   * @return The derived object
   * @see Curiously Recurring Template Pattern
   */
  T& asDerived() { return static_cast<T&>(*this); }
};

/**
 * @brief Fully evaluates a vector expression into an actual mfem::Vector
 * @param expr The expression to evaluate
 * @return The fully evaluated vector
 * @see VectorExpr::operator mfem::Vector
 */
template <typename T>
mfem::Vector evaluate(const VectorExpr<T>& expr)
{
  return expr;
}

/**
 * @brief Fully evaluates a vector expression into an actual mfem::Vector
 * @param expr The expression to evaluate
 * @param vec The vector to populate with the expression result
 */
template <typename T>
void evaluate(const VectorExpr<T>& expr, mfem::Vector& result)
{
  SLIC_ERROR_IF(expr.Size() != static_cast<std::size_t>(result.Size()),
                "Vector sizes in expression assignment must be equal");
  for (size_t i = 0; i < expr.Size(); i++) {
    result[i] = expr[i];
  }
}

/**
 * @brief Fully evaluates a vector expression into an actual mfem::Vector
 * @param expr The expression to evaluate
 * @param vec The vector to populate with the expression result
 * @param comm The MPI_Comm to use for parallel evaluation
 * @note The performance in a release build is comparable to the serial
 * implementation, though because this populates the vector on all ranks
 * the cost of the data copies may exceed any parallelization benefit
 * for the actual calculation
 */
template <typename T>
void evaluate(const VectorExpr<T>& expr, mfem::Vector& result, MPI_Comm comm)
{
  const auto SIZE = expr.Size();
  SLIC_ERROR_IF(SIZE != static_cast<std::size_t>(result.Size()), "Vector sizes in expression assignment must be equal");
  int num_procs = 0;
  int rank      = 0;
  MPI_Comm_size(comm, &num_procs);
  MPI_Comm_rank(comm, &rank);

  double* result_arr = result;

  // If the array size is not divisible by # elements, add one
  const long long per_proc = ((SIZE / num_procs) + (SIZE % num_procs != 0)) ? 1 : 0;

  // Truncate the number of elements for the last process
  const long long n_entries = (rank == num_procs - 1) ? SIZE - ((num_procs - 1) * per_proc) : per_proc;

  // Fill in this rank's segment of the vector
  for (long long i = 0; i < n_entries; i++) {
    result_arr[i + (per_proc * rank)] = expr[i + (per_proc * rank)];
  }

  // Transmit each segment of the vector to all the other processes
  for (int i = 0; i < num_procs; i++) {
    const long long n_ele = (i == num_procs - 1) ? SIZE - ((num_procs - 1) * per_proc) : per_proc;
    MPI_Bcast(&result_arr[per_proc * i], n_ele, MPI_DOUBLE, i, comm);
  }
}

}  // namespace serac

#endif
