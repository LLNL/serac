// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file inequality_constraint.hpp
 *
 * @brief Objects for defining inequality constraints
 */

#pragma once

#include "serac/numerics/functional/tensor.hpp"

//#define USE_SMOOTH_AL

namespace serac {

template <int dim>
class Constraint {
public:
  using vector_t = tensor<double, dim>;

  virtual ~Constraint() = default;

  virtual double   evaluate(const vector_t& x, double t) const                      = 0;
  virtual vector_t gradient(const vector_t& x, double t) const                      = 0;
  virtual vector_t hessianVec(const vector_t& x, const vector_t& w, double t) const = 0;
};

template <int dim>
class LevelSetPlane : public Constraint<dim> {
public:
  using vector_t = Constraint<dim>::vector_t;

  LevelSetPlane(vector_t center, vector_t normal, vector_t velocity = zero{})
      : center_(center), normal_(normal), velocity_(velocity)
  {
  }

  // positive means constraint is satisfied, i.e., c(x) >= 0
  double evaluate(const vector_t& x, double t) const override
  {
    auto c = center_ + t * velocity_;
    return dot(normal_, x - c);
  }

  vector_t gradient(const vector_t&, double) const override { return normal_; }

  vector_t hessianVec(const vector_t&, const vector_t&, double) const override { return zero{}; }

  template <typename Position>
  double operator()(Position x, double t) const
  {
    auto c = center_ + t * velocity_;
    return dot(normal_, x - c);
  }

private:
  const vector_t center_;
  const vector_t normal_;
  const vector_t velocity_;
};

template <int dim>
class LevelSetSphere : public Constraint<dim> {
public:
  using vector_t = Constraint<dim>::vector_t;

  LevelSetSphere(vector_t center, double radius, vector_t velocity = zero{})
      : center_(center), radius_(radius), velocity_(velocity)
  {
  }

  // positive means constraint is satisfied, i.e., c(x) >= 0
  double evaluate(const vector_t& x, double t) const override
  {
    auto c = center_ + t * velocity_;
    return norm(x - c) - radius_;
  }

  vector_t gradient(const vector_t& x, double t) const override
  {
    auto   c    = center_ + t * velocity_;
    double dist = norm(x - c);
    if (dist > DBL_EPSILON) {
      return (1.0 / dist) * (x - c);
    } else {
      return zero{};
    }
  }

  vector_t hessianVec(const vector_t& x, const vector_t& w, double t) const override
  {
    auto   c            = center_ + t * velocity_;
    double dist_squared = squared_norm(x - c);
    if (dist_squared > DBL_EPSILON) {
      double distInv           = 1.0 / std::sqrt(dist_squared);
      double dist_to_minus_1p5 = distInv / dist_squared;
      double factor            = dot(w, x - c);
      return distInv * w - dist_to_minus_1p5 * factor * (x - c);
    } else {
      return zero{};
    }
  }

  template <typename Position>
  auto operator()(Position x, double t) const
  {
    auto c = center_ + t * velocity_;
    return norm(x - c) - radius_;
  }

private:
  const vector_t center_;
  const double   radius_;
  const vector_t velocity_;
};

}  // namespace serac
