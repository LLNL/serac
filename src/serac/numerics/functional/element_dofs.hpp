#pragma once

#include <vector>

#include "mfem.hpp"

inline bool isH1(const mfem::FiniteElementSpace& fes)
{
  return (fes.FEColl()->GetContType() == mfem::FiniteElementCollection::CONTINUOUS);
}

inline bool isHcurl(const mfem::FiniteElementSpace& fes)
{
  return (fes.FEColl()->GetContType() == mfem::FiniteElementCollection::TANGENTIAL);
}

inline bool isDG(const mfem::FiniteElementSpace& fes)
{
  return (fes.FEColl()->GetContType() == mfem::FiniteElementCollection::DISCONTINUOUS);
}

enum class FaceType {BOUNDARY, INTERIOR};

struct DoF {

  // sam: I wanted to use a bitfield for this type, but a 10+ year-old GCC bug
  // makes it practically impossible to assign to bitfields without warnings/errors
  // with -Wconversion enabled, see: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=39170
  // So, we resort to masks and bitshifting instead.
  static constexpr uint64_t sign_mask = 0x8000'0000'0000'0000;
  static constexpr uint64_t orientation_mask = 0x7000'0000'0000'0000;
  static constexpr uint64_t index_mask = 0x0000'FFFF'FFFF'FFFF'FFFF;
  static constexpr uint64_t sign_shift = 63;
  static constexpr uint64_t orientation_shift = 60;
  static constexpr uint64_t index_shift = 0;

  // data layout is as follows (MSB to LSB): 
  // - 1 sign bit 
  // - 3 orientation bits
  // - 12 currently unused bits that may be used in the future
  // - 48 index bits
  //
  // all values are immutable unsigned integers
  const uint64_t bits;

  DoF(uint64_t index, uint64_t sign = 0, uint64_t orientation = 0) : 
    bits((sign & 0x1ULL << sign_shift) + ((orientation & 0x7ULL) << orientation_shift) + index)
  {}

  uint64_t sign() const { return ((bits & sign_mask) >> sign_shift); }
  uint64_t orientation() const { return ((bits & orientation_mask) >> orientation_shift); }
  uint64_t index() const { return (bits & index_mask); }

};

template <typename T>
struct Range {
  T* begin() { return ptr[0]; }
  T* end() { return ptr[1]; }
  T* ptr[2];
};

template <typename T>
struct Array2D {
  Array2D() : values{}, dim{} {};

  Array2D(uint64_t m, uint64_t n) : values(m * n, 0), dim{m, n} {};
  Array2D(std::vector<T>&& data, uint64_t m, uint64_t n) : values(data), dim{m, n} {};
  Range<T>       operator()(uint64_t i) { return Range<T>{&values[i * dim[1]], &values[(i + 1) * dim[1]]}; }
  Range<const T> operator()(uint64_t i) const { return Range<const T>{&values[i * dim[1]], &values[(i + 1) * dim[1]]}; }
  T&             operator()(uint64_t i, uint64_t j) { return values[i * dim[1] + j]; }
  const T&       operator()(uint64_t i, uint64_t j) const { return values[i * dim[1] + j]; }

  // these overloads exist to mitigate the excessive `static_cast`ing
  // necessary to coexist with mfem's convention where everything is an `int`
  Array2D(int m, int n) : values(uint64_t(m) * uint64_t(n), 0), dim{uint64_t(m), uint64_t(n)} {}
  Array2D(std::vector<T>&& data, int m, int n) : values(data), dim{uint64_t(m), uint64_t(n)} {}
  Range<T>       operator()(int i) { return Range<T>{&values[uint64_t(i) * dim[1]], &values[uint64_t(i + 1) * dim[1]]}; }
  Range<const T> operator()(int i) const { return Range<const T>{&values[uint64_t(i) * dim[1]], &values[uint64_t(i + 1) * dim[1]]}; }
  T&             operator()(int i, int j) { return values[uint64_t(i) * dim[1] + uint64_t(j)]; }
  const T&       operator()(int i, int j) const { return values[uint64_t(i) * dim[1] + uint64_t(j)]; }

  std::vector<T> values;
  uint64_t       dim[2];
};

struct ElementDofs {
  ElementDofs() {}
  ElementDofs(const mfem::FiniteElementSpace* fes, mfem::Geometry::Type elem_geom);
  ElementDofs(const mfem::FiniteElementSpace* fes, mfem::Geometry::Type face_geom, FaceType type);

  uint64_t ESize();
  uint64_t LSize();
  void Gather(const mfem::Vector & L_vector, mfem::Vector & E_vector) const;
  void ScatterAdd(const mfem::Vector & E_vector, mfem::Vector & L_vector) const;

  uint64_t esize, lsize, components;
  Array2D<DoF> dof_info;
};

Array2D<DoF> GetElementDofs(mfem::FiniteElementSpace* fes, mfem::Geometry::Type geom);
Array2D<DoF> GetFaceDofs(mfem::FiniteElementSpace* fes, mfem::Geometry::Type face_geom, FaceType type);
