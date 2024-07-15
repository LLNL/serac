// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file solid_mechanics.hpp
 *
 * @brief An object for representing inequality constraints
 */

#pragma once

#include "mfem.hpp"

#include "serac/numerics/stdfunction_operator.hpp"
#include "serac/numerics/functional/shape_aware_functional.hpp"

//#define USE_SMOOTH_AL

namespace serac {

template <int dim>
struct NodalFriction 
{

  double mu_;
  double v_crit_;
  std::array<double, dim> v;

  NodalFriction(double mu, double v_crit, std::array<double, dim> referenceV) : mu_(mu), v_crit_(v_crit), v(referenceV)
  {
  }

  /*
  double quasiVariationalPotential(const mfem::Vector& x, const mfem::Vector& x_prev,
                                   double dt) const
  {
    if (dt == 0.0 || mu_ == 0.0) {
      return 0.0;
    }

    mfem::Vector v_perp = x;
    v_perp = 0.0;

    for (int i=0; i < dim; ++i) v_perp[i] = (x[i] - x_prev[i]) / dt - v[static_cast<size_t>(i)];

    double vv = 0.0; for (int i=0; i < dim; ++i) vv += v_perp[i]*v_perp[i];

    if (vv < v_crit_*v_crit_) {
      return 0.5 * dt * mu_ * vv / v_crit_;
    } else {
      return dt * mu_ * std::sqrt(vv);
    }
  }
  */

  auto quasiVariationalResidual(const mfem::Vector& x, const mfem::Vector& x_prev, const mfem::Vector& normal,
                                        double dt) const
  {
    mfem::Vector v_perp = x; v_perp = 0.0;
    if (dt == 0.0 || mu_ == 0.0) {
      return std::make_pair(v_perp, false);
    }

    if (std::abs(normal.Norml2() - 1.0) > 1e-9) {
      printf("bad norm\n");
    }

    for (int i=0; i < dim; ++i) v_perp[i] = (x[i] - x_prev[i]) / dt - v[static_cast<size_t>(i)];
    double vn = 0.0; for (int i=0; i < dim; ++i) vn += v_perp[i]*normal[i];
    for (int i=0; i < dim; ++i) v_perp[i] -= normal[i] * vn;
    double vv = 0.0; for (int i=0; i < dim; ++i) vv += v_perp[i]*v_perp[i];

    if (vv == 0.0) {
      v_perp = 0.0;
      return std::make_pair(v_perp, false);
    }

    bool is_fast = false;

    if (vv < v_crit_*v_crit_) {
      for (int i=0; i < dim; ++i) v_perp[i] = mu_ * v_perp[i] / v_crit_;
    } else {
      is_fast = true;
      double sqrtvv = std::sqrt(vv);
      if (sqrtvv != sqrtvv || sqrtvv == 0.0) {
        v_perp = 0.0;
      } else {
        for (int i=0; i < dim; ++i) v_perp[i] = mu_ * (v_perp[i] / sqrtvv);
      }
    }

    // mu / dt * I / sqrtvv * d(v_perp) * P 
    // mu / dt * vp * (-1) vp / sqrtvv^3

    return std::make_pair(v_perp, is_fast);
  }

  auto quasiVariationalHessVec(const mfem::Vector& x, const mfem::Vector& x_prev, const mfem::Vector& normal, const mfem::Vector& w,
                                       double dt) const
  {
    mfem::Vector Hw = x; Hw = 0.0;
    mfem::Vector v_perp = x; v_perp = 0.0;
    mfem::Vector w_perp = w;
    if (dt == 0.0 || mu_ == 0.0) {
      return std::make_pair(Hw, false);
    }

    if (std::abs(normal.Norml2() - 1.0) > 1e-9) {
      printf("bad norm\n");
      exit(1);
    }

    for (int i=0; i < dim; ++i) v_perp[i] = (x[i] - x_prev[i]) / dt - v[static_cast<size_t>(i)];
    double vn = 0.0; for (int i=0; i < dim; ++i) vn += v_perp[i]*normal[i];
    for (int i=0; i < dim; ++i) v_perp[i] -= normal[i] * vn;
    double vv = 0.0; for (int i=0; i < dim; ++i) vv += v_perp[i]*v_perp[i];

    //double wn = 0.0; for (int i=0; i < dim; ++i) wn += w[i]*normal[i];
    //for (int i=0; i < dim; ++i) w_perp[i] = w[i] * wn;

    if (vv == 0.0) {
      return std::make_pair(Hw, false);
    }

    bool fast = true;

    if (vv < v_crit_*v_crit_) {
      //printf("slow\n");
      fast = false;
      for (int i=0; i < dim; ++i) Hw[i] = (mu_ / (dt * v_crit_)) * w_perp[i];
    } else {
      double sqrtvv = std::sqrt(vv);
      if (sqrtvv != sqrtvv || sqrtvv == 0.0) {
        Hw = 0.0;
      } else {
        for (int i=0; i < dim; ++i) v_perp[i] /= sqrtvv;
        double vw = 0.0; for (int i=0; i < dim; ++i) vw += v_perp[i] * w_perp[i];
        for (int i=0; i < dim; ++i) Hw[i] = (mu_ / (dt * sqrtvv)) * (w_perp[i] - vw * v_perp[i]);
      }
    }

    double Hvn = 0.0; for (int i=0; i < dim; ++i) Hvn += Hw[i]*normal[i];
    for (int i=0; i < dim; ++i) Hw[i] -= normal[i] * Hvn;
    
    /*
    double eps = 1e-8;
    mfem::Vector Hw_fd = x; Hw_fd = 0.0;
    auto r = quasiVariationalResidual(x, x_prev, normal, dt);


     // drj_dxi
    for (int i=0; i < dim; ++i) {
      mfem::Vector xp = x;
      xp[i] += eps;
      auto rplus = quasiVariationalResidual(xp, x_prev, normal, dt);
      // dr[j]/dx[i]*w[i];
      for (int j=0; j < dim; ++j) {
        Hw_fd[j] += (rplus[j] - r[j]) * w[i] / eps;
      }
    }
    double error = 0.0; for (int i=0; i < dim; ++i) error += std::abs(Hw[i] - Hw_fd[i]);
    if (false && error > 1e-5) {
      if (vv < v_crit_*v_crit_) std::cout << "case slow\n";
      else std::cout << "case fast\n";
      std::cout << "Hw  , = " << Hw[0] << " " << Hw[1] << " " << Hw[2] << std::endl;
      std::cout << "Hwfd, = " << Hw_fd[0] << " " << Hw_fd[1] << " " << Hw_fd[2] << std::endl;
      std::cout << "error = " << error << std::endl;
    }
    */

    return std::make_pair(Hw, fast);
  }

};


struct LevelSet {
  virtual ~LevelSet() {}
  // positive means constraint is satisfied, i.e., c(x) >= 0
  virtual double evaluate(const mfem::Vector& x, double t) const = 0;

  virtual mfem::Vector gradient(const mfem::Vector& x, double t) const = 0;

  virtual mfem::Vector hessVec(const mfem::Vector& x, const mfem::Vector&, double t) const = 0;
};

template <int dim>
struct LevelSetPlane : public LevelSet { // hard codes as a y plane for now

  LevelSetPlane(std::array<double,dim> c_, std::array<double,dim> n_) : c(c_), n(n_)
  {
  }

  std::array<double,dim> c;
  std::array<double,dim> n;

  // positive means constraint is satisfied, i.e., c(x) >= 0
  double evaluate(const mfem::Vector& xyz, double) const override
  { 
    double val = 0.0;
    for (int i=0; i < dim; ++i) {
      auto i_size_t = static_cast<size_t>(i);
      val += n[i_size_t] * (xyz[i] - c[i_size_t]);
    }
    return val;
  }

  mfem::Vector gradient(const mfem::Vector& xyz, double) const override
  {
    mfem::Vector grad = xyz;
    for (int i=0; i < dim; ++i) {
      auto i_size_t = static_cast<size_t>(i);
      grad[i] = n[i_size_t];
    }
    return grad;
  }

  mfem::Vector hessVec(const mfem::Vector& xyz, const mfem::Vector& /*w*/, double) const override
  {
    mfem::Vector hv = xyz;
    hv              = 0.0;
    return hv;
  }
};


template <int dim>
struct LevelSetSphere : public LevelSet {

  LevelSetSphere(std::array<double,dim> c_, double r_, std::array<double,dim> v_) : center(c_), r(r_), v(v_) {}

  const std::array<double,dim> center;
  const double r;
  const std::array<double,dim> v;

  double sq(double a) const { return a*a; }

  double distSquared(const mfem::Vector& xyz, const std::array<double,dim>& c) const
  {
    double dist_squared = 0.0;
    for (int i=0; i < dim; ++i) {
      dist_squared += sq(xyz[i] - c[static_cast<size_t>(i)]);
    }
    return dist_squared;
  }

  // positive means constraint is satisfied, i.e., c(x) >= 0
  double evaluate(const mfem::Vector& xyz, double t) const override
  { 
    auto c = center;
    for (size_t i=0; i<dim; ++i) {
      c[i] += t * v[i];
    }
    return std::sqrt(distSquared(xyz, c)) - r;
  }

  mfem::Vector gradient(const mfem::Vector& x, double t) const override
  {
    auto c = center;
    for (size_t i=0; i<dim; ++i) {
      c[i] += t * v[i];
    }
    mfem::Vector grad = x; grad = 0.0;
    double dist_squared = distSquared(x, c);
    if (dist_squared != 0.0) {
      double distInv = 1.0 / std::sqrt(dist_squared);
      for (int i=0; i < dim; ++i) {
        grad[i] = (x[i] - c[static_cast<size_t>(i)]) * distInv;
      }
    }
    return grad;
  }

  mfem::Vector hessVec(const mfem::Vector& x, const mfem::Vector& w, double t) const override
  {
    auto c = center;
    for (size_t i=0; i<dim; ++i) {
      c[i] += t * v[i];
    }
    mfem::Vector hv = x; hv = 0.0;
    double dist_squared = distSquared(x, c);
    if (dist_squared != 0.0) {
      double distInv = 1.0 / std::sqrt(dist_squared);
      double dist_to_minus_1p5 = distInv / dist_squared;
      double factor = 0.0;
      for (int i=0; i < dim; ++i) {
        factor += w[i] * (x[i] - c[static_cast<size_t>(i)]);
      }
      for (int i=0; i < dim; ++i) {
        hv[i] = distInv * w[i] - dist_to_minus_1p5 * factor * (x[i] - c[static_cast<size_t>(i)]);
      }
    }

    return hv;
  }

};

template <int dim>
struct ConstView {
  ConstView(const mfem::Vector& v_) : v(v_), numNodes_(v.Size() / dim) {}

  int getIndex(int i, int j) const { return mfem::Ordering::Map<mfem::Ordering::byNODES>(numNodes_, dim, i, j); }

  const double& operator[](int i) const { return v[i]; }
  const double& operator()(int i) const { return v[i]; }
  const double& operator()(int i, int j) const { return v[getIndex(i, j)]; }

  int numNodes() const { return numNodes_; }

private:
  const mfem::Vector& v;
  int                 numNodes_;
};

template <int dim>
struct View {
  View(mfem::Vector& v_) : v(v_), numNodes_(v.Size() / dim) {}

  int getIndex(int i, int j) const { return mfem::Ordering::Map<mfem::Ordering::byNODES>(numNodes_, dim, i, j); }

  double&       operator[](int i) { return v[i]; }
  double&       operator()(int i) { return v[i]; }
  const double& operator[](int i) const { return v[i]; }
  const double& operator()(int i) const { return v[i]; }

  double&       operator()(int i, int j) { return v[getIndex(i, j)]; }
  const double& operator()(int i, int j) const { return v[getIndex(i, j)]; }

  int numNodes() const { return numNodes_; }

private:
  mfem::Vector& v;
  int           numNodes_;
};

template <int order, int dim>
struct InequalityConstraint {
  InequalityConstraint(std::unique_ptr<LevelSet> levelSet, std::unique_ptr<NodalFriction<dim>> friction,
                       std::string physics_name, std::string mesh_tag,
                       double initial_penalty)
      : levelSet_(std::move(levelSet)),
        friction_(std::move(friction)),
        initial_penalty_(initial_penalty),
        constraint_(StateManager::newState(H1<order, 1>{}, detail::addPrefix(physics_name, "constraint"), mesh_tag)),
        constraint_multiplier_(
            StateManager::newDual(H1<order, 1>{}, detail::addPrefix(physics_name, "constraint_multiplier"), mesh_tag)),
        constraint_penalty_(
            StateManager::newState(H1<order, 1>{}, detail::addPrefix(physics_name, "constraint_penalty"), mesh_tag)),
        constraint_ncp_error_(
            StateManager::newState(H1<order, 1>{}, detail::addPrefix(physics_name, "constraint_ncp_error"), mesh_tag)),
        constraint_diagonal_stiffness_(StateManager::newState(
            H1<order, dim * dim>{}, detail::addPrefix(physics_name, "constraint_diagonal_stiffness"), mesh_tag))
  {
    reset();
  }

  void reset()
  {
    constraint_            = 0.0;
    constraint_multiplier_ = 0.1;
    constraint_penalty_    = initial_penalty_;
    constraint_ncp_error_  = 0.1 * std::numeric_limits<double>::max();
  }

  void outputStateToDisk() const
  {
    StateManager::updateState(constraint_);
    StateManager::updateDual(constraint_multiplier_);
    StateManager::updateState(constraint_penalty_);
    StateManager::updateState(constraint_ncp_error_);
    StateManager::updateState(constraint_diagonal_stiffness_);
  }


  double phi(double t) const {
    if (t <= 0.5 ) {
      return -t + 0.5 * t * t;  // -0.5 + 0.5 * 0.5 * 0.5 = -4/8 + 1/8 = -3/8
    } else {
      return -0.25 * std::log(2*t) - 3.0 / 8.0;
    }
  }

  double dphi(double t) const {
    if (t <= 0.5 ) {
      return -1.0 + t; // -1 + t = -0.5
    } else {
      return -0.25 / t; //-1/4t  *  + t = -0.5
    }
  }

  double ddphi(double t) const {
    if (t <= 0.5 ) {
      return 1.0;
    } else {
      return 0.25 / (t*t);
    }
  }

  void sumConstraintResidual(const FiniteElementVector& x_current, const FiniteElementVector& x_recent, const FiniteElementVector& x_previous, double time, double dt, mfem::Vector& res)
  {
    View<1>        constraint(constraint_);
    View<1>        constraint_multiplier(constraint_multiplier_);
    View<1>        constraint_penalty(constraint_penalty_);
    ConstView<dim> x(x_current);
    ConstView<dim> x_recnt(x_recent);
    ConstView<dim> x_prev(x_previous);
    View<dim>      residual(res);

    const int num_nodes = x.numNodes();
    SLIC_ERROR_ROOT_IF(num_nodes != constraint_.Size(), "Constraint size does not match system size.");
    SLIC_ERROR_ROOT_IF(num_nodes != x_prev.numNodes(), "Constraint size does not match system size.");

    //size_t active_count = 0;
    //size_t fast_count = 0;

    mfem::Vector coord(dim);
    mfem::Vector coord_rcnt(dim);
    mfem::Vector coord_prev(dim);
    for (int n = 0; n < num_nodes; ++n) {
      for (int i = 0; i < dim; ++i) {
        coord[i] = x(n, i);
        coord_rcnt[i] = x_recnt(n, i);
        coord_prev[i] = x_prev(n, i);
      }

      const double c   = levelSet_->evaluate(coord, time);
      constraint[n]    = c;
      const double lam = constraint_multiplier[n];
      const double k   = constraint_penalty[n];

#if !defined(USE_SMOOTH_AL)
      if (lam >= k * c) {
        //++active_count;
#else
      //if ( k * c  < 0.5 * lam) ++activeCount;
#endif
        // objective = lam * lam / k * phi( k * c / lam )
        const mfem::Vector gradC = levelSet_->gradient(coord, time);
        
#if defined(USE_SMOOTH_AL)
        double phip = dphi(k * c / lam);
#endif

        for (int i = 0; i < dim; ++i) {
#if defined(USE_SMOOTH_AL)
          residual(n, i) += lam * phip * gradC[i];
#else
          residual(n, i) += gradC[i] * (-lam + k * c);
#endif
        }
#if !defined(USE_SMOOTH_AL)
      }
#endif
      if (lam > 0.0) {
        const mfem::Vector gradCOld = levelSet_->gradient(coord_rcnt, time);
        //std::cout << "prev coord = " << gradCOld.Norml2() << std::endl;
        auto [qvRes, is_fast] = friction_->quasiVariationalResidual(coord, coord_prev, gradCOld, dt);
        //if (is_fast) fast_count++;
        for (int i = 0; i < dim; ++i) {
          residual(n, i) += lam * qvRes[i];
        }
      }

    }
    // std::cout << "num active, fast constraints res = " << active_count << " " << fast_count << std::endl;
  }


  std::unique_ptr<mfem::HypreParMatrix> sumConstraintJacobian(const FiniteElementVector& x_current,
                                                              const FiniteElementVector& x_recent,
                                                              const FiniteElementVector& x_previous,
                                                              double time,
                                                              double dt,
                                                              std::unique_ptr<mfem::HypreParMatrix> J)
  {
    constraint_diagonal_stiffness_ = 0.0;

    View<1>         constraint(constraint_);
    View<1>         constraint_multiplier(constraint_multiplier_);
    View<1>         constraint_penalty(constraint_penalty_);
    ConstView<dim>  x(x_current);
    ConstView<dim> x_recnt(x_recent);
    ConstView<dim>  x_prev(x_previous);
    View<dim * dim> constraint_diagonal_stiffness(constraint_diagonal_stiffness_);

    const int numNodes = x.numNodes();
    SLIC_ERROR_ROOT_IF(numNodes != constraint_.Size(), "Constraint size does not match system size.");

    mfem::Vector coord(dim);  // switch to stack vectors eventually
    mfem::Vector coord_rcnt(dim);
    mfem::Vector coord_prev(dim);  // switch to stack vectors eventually
    mfem::Vector xyz_dirs(dim);

    //size_t active_count = 0;
    //size_t fast_count = 0;

    for (int n = 0; n < numNodes; ++n) {
      for (int i = 0; i < dim; ++i) {
        coord[i] = x(n, i);
        coord_rcnt[i] = x_recnt(n, i);
        coord_prev[i] = x_prev(n, i);
      }
      const double c   = levelSet_->evaluate(coord, time);
      constraint[n]    = c;
      const double lam = constraint_multiplier[n];
      const double k   = constraint_penalty[n];
#if defined(USE_SMOOTH_AL)
        double phipp = ddphi(k * c / lam);
        double phip = dphi(k * c / lam);
#else
      if (lam >= k * c) {
        //++active_count;
#endif

        const mfem::Vector gradC = levelSet_->gradient(coord, time);
        for (int i = 0; i < dim; ++i) {
          xyz_dirs                 = 0.0;
          xyz_dirs[i]              = 1.0;
          const mfem::Vector hessI = levelSet_->hessVec(coord, xyz_dirs, time);
          for (int j = 0; j < dim; ++j) {
#if defined(USE_SMOOTH_AL)
            constraint_diagonal_stiffness(n, dim * i + j) += k * gradC[i] * phipp * gradC[j] + k * hessI[j] * phip * phip; // + hessI[j] * (-lam + k * c);
#else
            constraint_diagonal_stiffness(n, dim * i + j) += k * gradC[i] * gradC[j] + hessI[j] * (-lam + k * c);
#endif
          }
        }
#if !defined(USE_SMOOTH_AL)
      }
#endif
      if (lam > 0.0) {
        const mfem::Vector gradCOld = levelSet_->gradient(coord_rcnt, time);
        for (int i = 0; i < dim; ++i) {
          xyz_dirs                 = 0.0;
          xyz_dirs[i]              = 1.0;
          auto [qvHessI, is_fast] = friction_->quasiVariationalHessVec(coord, coord_prev, gradCOld, xyz_dirs, dt);
          //if (is_fast) fast_count++;
          for (int j = 0; j < dim; ++j) {
            constraint_diagonal_stiffness(n, dim * i + j) += lam * qvHessI[j];
          }
        }
      }
    }

    hypre_ParCSRMatrix* J_hype(*J);

    auto*       Jdiag_data = hypre_CSRMatrixData(J_hype->diag);
    const auto* Jdiag_i    = hypre_CSRMatrixI(J_hype->diag);
    const auto* Jdiag_j    = hypre_CSRMatrixJ(J_hype->diag);

    using array    = std::array<int, dim>;
    using arrayInt = typename array::size_type;

    std::array<int, dim> nodalCols;
    for (int n = 0; n < numNodes; ++n) {
      for (int i = 0; i < dim; ++i) {
        nodalCols[static_cast<arrayInt>(i)] = x.getIndex(n, i);
      }

      for (int i = 0; i < dim; ++i) {
        int  row      = x.getIndex(n, i);
        auto rowStart = Jdiag_i[row];
        auto rowEnd   = Jdiag_i[row + 1];
        for (auto colInd = rowStart; colInd < rowEnd; ++colInd) {
          int   col = Jdiag_j[colInd];
          auto& val = Jdiag_data[colInd];
          for (int j = 0; j < dim; ++j) {
            if (col == nodalCols[static_cast<arrayInt>(j)]) {
              val += constraint_diagonal_stiffness(n, dim * i + j);
            }
          }
        }
      }
    }

    //std::cout << "num active, fast constraints hv = " << active_count << " " << fast_count/3 << std::endl;

    return J;
  }

  double updateMultipliers(const FiniteElementVector& x_current, double time)
  {
    double target_decrease_factor = 0.75;

    auto fischer_burmeister_ncp_error = [](double c, double lam, double k) {
      double ck = c * k;
      return std::sqrt(ck * ck + lam * lam) - ck - lam;
    };

    View<1>        constraint(constraint_);
    View<1>        constraint_multiplier(constraint_multiplier_);
    View<1>        constraint_penalty(constraint_penalty_);
    View<1>        constraint_ncp_error(constraint_ncp_error_);
    ConstView<dim> x(x_current);

    const int numNodes = x.numNodes();
    SLIC_ERROR_ROOT_IF(numNodes != constraint_.Size(), "Constraint size does not match system size.");

    for (int n = 0; n < numNodes; ++n) {
      mfem::Vector currentCoords(dim);
      for (int i = 0; i < dim; ++i) {
        currentCoords[i] = x(n, i);
      }
      const double c = levelSet_->evaluate(currentCoords, time);
      constraint[n]  = c;

      double lam = constraint_multiplier[n];
      const double k   = constraint_penalty[n];
      // update multiplier
#if defined(USE_SMOOTH_AL)
      lam *= -dphi( c * k / lam );
#else
      lam = std::max(lam - k * c, 0.0);
#endif
      constraint_multiplier[n] = lam;

      double oldError         = constraint_ncp_error[n];
      double newError         = std::abs(fischer_burmeister_ncp_error(c, lam, k));
      constraint_ncp_error[n] = newError;

      bool poorProgress = newError > target_decrease_factor * oldError;

      if (poorProgress) constraint_penalty[n] *= 1.0; //0.1;//1;
    }

    //std::cout << "lam norm = " << constraint_multiplier_.Norml2() << std::endl;
    std::cout << "ncp error = " << constraint_ncp_error_.Norml2() << std::endl;

    return constraint_ncp_error_.Norml2();

    // ncpError = np.abs( alObjective.ncp(x) )
    //# check if each constraint is making progress, or if they are already small relative to the specificed constraint
    // tolerances poorProgress = ncpError > np.maximum(alSettings.target_constraint_decrease_factor * ncpErrorOld, 10 *
    // alSettings.tol / np.sqrt(len(ncpError)))
  }

protected:
  const std::unique_ptr<LevelSet> levelSet_;
  const std::unique_ptr<NodalFriction<3>> friction_;
  double initial_penalty_;

  FiniteElementState constraint_;
  FiniteElementDual  constraint_multiplier_;
  FiniteElementState constraint_penalty_;
  FiniteElementState constraint_ncp_error_;
  FiniteElementState constraint_diagonal_stiffness_;
};

}  // namespace serac