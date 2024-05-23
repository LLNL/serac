// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/numerics/equation_solver.hpp"
#include <iomanip>
#include <sstream>
#include <ios>
#include <iostream>

#include "serac/infrastructure/logger.hpp"
#include "serac/infrastructure/terminator.hpp"
#include "serac/serac_config.hpp"

namespace serac {

class NewtonSolver : public mfem::NewtonSolver {
protected:
  mutable mfem::Vector   x0;
  NonlinearSolverOptions nonlinear_options;

public:
  NewtonSolver(const NonlinearSolverOptions& nonlinear_opts) : nonlinear_options(nonlinear_opts) {}

#ifdef MFEM_USE_MPI
  NewtonSolver(MPI_Comm comm_, const NonlinearSolverOptions& nonlinear_opts)
      : mfem::NewtonSolver(comm_), nonlinear_options(nonlinear_opts)
  {
  }
#endif

  void Mult(const mfem::Vector&, mfem::Vector& x) const
  {
    MFEM_ASSERT(oper != NULL, "the Operator is not set (use SetOperator).");
    MFEM_ASSERT(prec != NULL, "the Solver is not set (use SetSolver).");

    using real_t = mfem::real_t;

    real_t norm, norm_goal;
    oper->Mult(x, r);

    norm = initial_norm = Norm(r);
    if (print_options.first_and_last && !print_options.iterations) {
      mfem::out << "Newton iteration " << std::setw(2) << 0 << " : ||r|| = " << norm << "...\n";
    }

    norm_goal            = std::max(rel_tol * initial_norm, abs_tol);
    prec->iterative_mode = false;

    int it = 0;
    for (; true; it++) {
      MFEM_ASSERT(IsFinite(norm), "norm = " << norm);
      if (print_options.iterations) {
        mfem::out << "Newton iteration " << std::setw(2) << it << " : ||r|| = " << norm;
        if (it > 0) {
          mfem::out << ", ||r||/||r_0|| = " << norm / initial_norm;
        }
        mfem::out << '\n';
      }

      if (norm <= norm_goal && it >= nonlinear_options.min_iterations) {
        converged = true;
        break;
      }

      if (it >= max_iter) {
        converged = false;
        break;
      }

      real_t norm_nm1 = norm;

      grad = &oper->GetGradient(x);
      prec->SetOperator(*grad);

      prec->Mult(r, c);  // c = [DF(x_i)]^{-1} [F(x_i)-b]

      // there must be a better way to do this?
      x0.SetSize(x.Size());
      x0 = 0.0;
      x0.Add(1.0, x);

      real_t c_scale = 1.0;
      add(x0, -c_scale, c, x);
      oper->Mult(x, r);
      norm = Norm(r);

      const int               max_ls_iters = nonlinear_options.max_line_search_iterations;
      static constexpr real_t reduction    = 0.5;

      const double cMagnitudeInR         = std::abs(Dot(c, r)) / norm_nm1;
      const double sufficientDegreeParam = 0.0;  // 1e-15;
      auto         is_improved           = [=](real_t currentNorm, real_t c_scale) {
        return currentNorm < norm_nm1 - sufficientDegreeParam * c_scale * cMagnitudeInR;
      };

      // back-track linesearch
      int ls_iter     = 0;
      int ls_iter_sum = 0;
      for (; !is_improved(norm, c_scale) && ls_iter < max_ls_iters; ++ls_iter, ++ls_iter_sum) {
        c_scale *= reduction;
        add(x0, -c_scale, c, x);
        oper->Mult(x, r);
        norm = Norm(r);
      }

      // try the opposite direction and linesearch back from there
      if (ls_iter == max_ls_iters && !is_improved(norm, c_scale)) {
        c_scale = 1.0;
        add(x0, c_scale, c, x);
        oper->Mult(x, r);
        norm = Norm(r);

        ls_iter = 0;
        for (; !is_improved(norm, c_scale) && ls_iter < max_ls_iters; ++ls_iter, ++ls_iter_sum) {
          c_scale *= reduction;
          add(x0, c_scale, c, x);
          oper->Mult(x, r);
          norm = Norm(r);
        }

        // ok, the opposite direction was also terrible, lets go back, cut in half 1 last time and accept it hoping for
        // the best
        if (ls_iter == max_ls_iters && !is_improved(norm, c_scale)) {
          ++ls_iter_sum;
          c_scale *= reduction;
          add(x0, -c_scale, c, x);
          oper->Mult(x, r);
          norm = Norm(r);
        }
      }

      if (ls_iter_sum) {
        if (print_options.iterations) {
          mfem::out << "Number of line search steps taken = : " << ls_iter_sum << std::endl;
        }
        if (print_options.warnings && (ls_iter_sum == 2 * max_ls_iters + 1)) {
          mfem::out << "The maximum number of line search cut back have occurred, the resulting residual may not have "
                       "decreased. "
                    << std::endl;
        }
      }
    }

    final_iter = it;
    final_norm = norm;

    if (print_options.summary || (!converged && print_options.warnings) || print_options.first_and_last) {
      mfem::out << "Newton: Number of iterations: " << final_iter << '\n' << "   ||r|| = " << final_norm << '\n';
    }
    if (!converged && (print_options.summary || print_options.warnings)) {
      mfem::out << "Newton: No convergence!\n";
    }
  }
};

class Nesterov : public mfem::NewtonSolver {
protected:
  mutable mfem::Vector v;
  mutable mfem::Vector a;
  mutable mfem::Vector xPred;
  mutable mfem::Vector Ks;
  mutable mfem::Vector rOld;
  mutable mfem::Vector rPred;
  mutable double       dt;

public:
  Nesterov() {}

#ifdef MFEM_USE_MPI
  Nesterov(MPI_Comm comm_) : mfem::NewtonSolver(comm_) {}
#endif

  void Mult(const mfem::Vector&, mfem::Vector& x) const
  {
    MFEM_ASSERT(oper != NULL, "the Operator is not set (use SetOperator).");
    MFEM_ASSERT(prec != NULL, "the Solver is not set (use SetSolver).");

    using real_t = mfem::real_t;

    real_t norm, norm_goal;
    oper->Mult(x, r);

    mfem::out << "initial residual norm = " << Norm(r) << std::endl;

    norm = initial_norm = Norm(r);
    norm_goal           = std::max(rel_tol * initial_norm, abs_tol);
    if (print_options.first_and_last && !print_options.iterations) {
      mfem::out << "Newton iteration " << std::setw(2) << 0 << " : ||r|| = " << norm << "...\n";
    }
    prec->iterative_mode = false;

    // local arrays
    v.SetSize(x.Size());
    v = 0.0;
    a.SetSize(x.Size());
    a = 0.0;
    xPred.SetSize(x.Size());
    xPred = 0.0;
    Ks.SetSize(x.Size());
    Ks = 0.0;
    rPred.SetSize(x.Size());
    rPred = 0.0;
    rOld.SetSize(x.Size());
    rOld = 0.0;

    int it = 0;
    for (; true; it++) {
      MFEM_ASSERT(IsFinite(norm), "norm = " << norm);
      if (print_options.iterations) {
        mfem::out << "Newton iteration " << std::setw(2) << it << " : ||r|| = " << norm;
        if (it > 0) {
          mfem::out << ", ||r||/||r_0|| = " << norm / initial_norm;
        }
        mfem::out << '\n';
      }

      if (norm <= norm_goal) {
        converged = true;
        break;
      } else if (it >= max_iter) {
        converged = false;
        break;
      }

      auto K = &oper->GetGradient(x);
      if (it == 0) {
        printf("setting preconditioner\n");
        prec->SetOperator(*K);
      }

      std::cout << "r a " << Norm(r) << std::endl;
      prec->Mult(r, c);  // c = [DF(x_i)]^{-1} [F(x_i)-b]
      std::cout << "c a " << Norm(c) << std::endl;

      real_t c_scale = 0.9;  // 0.1; //1.0; //1e-2;//1.0;
      K->Mult(c, Ks);
      double alpha = Dot(Ks, c) / Dot(c, c);
      c_scale      = 1.0 / (alpha * std::sqrt(Dot(c, c)));
      std::cout << "initial c scale = " << c_scale << std::endl;

      rOld           = r;
      double normOld = norm;

      add(x, -c_scale, c, xPred);
      add(rOld, -c_scale, Ks, rPred);

      double normPred = Norm(rPred);

      oper->Mult(xPred, r);
      norm = Norm(r);

      std::cout << "nrm , pre = " << norm - normOld << " " << normPred - normOld << std::endl;

      static constexpr int    max_ls_iters = 30;
      static constexpr real_t reduction    = 0.5;

      auto is_progress = [=](real_t currentNorm, real_t) {
        return ((currentNorm < 1.001 * normPred) || currentNorm < normOld);
      };

      // back-track linesearch
      int ls_iter     = 0;
      int ls_iter_sum = 0;
      for (; !is_progress(norm, c_scale) && ls_iter < max_ls_iters; ++ls_iter, ++ls_iter_sum) {
        c_scale *= reduction;
        add(x, -c_scale, c, xPred);
        add(rOld, -c_scale, Ks, rPred);
        normPred = Norm(rPred);
        oper->Mult(xPred, r);
        norm = Norm(r);
        std::cout << "nrm , pre = " << norm - normOld << " " << normPred - normOld << std::endl;
      }

      x = xPred;

      if (ls_iter_sum) {
        if (print_options.summary || (!converged && print_options.warnings) || print_options.first_and_last) {
          mfem::out << "Number of line search steps taken = : " << ls_iter_sum << std::endl;
        }
      }
    }

    final_iter = it;
    final_norm = norm;

    if (print_options.summary || (!converged && print_options.warnings) || print_options.first_and_last) {
      mfem::out << "Newton: Number of iterations: " << final_iter << '\n' << "   ||r|| = " << final_norm << '\n';
    }
    if (!converged && (print_options.summary || print_options.warnings)) {
      mfem::out << "Newton: No convergence!\n";
    }
  }
};

struct TrustRegionSettings {
  double cgTol                  = 1e-8;
  size_t maxCgIterations        = 5000;
  size_t maxCumulativeIteration = 1;
  double min_tr_size            = 1e-9;
  double t1                     = 0.25;
  double t2                     = 1.75;
  double eta1                   = 1e-9;
  double eta2                   = 0.1;
  double eta3                   = 0.6;
};

struct TrustRegionResults {
  TrustRegionResults(int size)
  {
    z.SetSize(size);
    d.SetSize(size);
    Pr.SetSize(size);
    Hd.SetSize(size);
    cauchyPoint.SetSize(size);
  }

  void reset()
  {
    z           = 0.0;
    cauchyPoint = 0.0;
  }

  enum class Status
  {
    Interior,
    NegativeCurvature,
    OnBoundary,
  };

  mfem::Vector z;   // step direction
  mfem::Vector d;   // incrementalCG direction
  mfem::Vector Pr;  // preconditioned residual
  mfem::Vector Hd;  // action of hessian on direction d
  mfem::Vector cauchyPoint;
  Status       interiorStatus    = Status::Interior;
  size_t       cgIterationsCount = 0;
};

class TrustRegion : public mfem::NewtonSolver {
protected:
  mutable mfem::Vector xPred;
  mutable mfem::Vector rPred;
  mutable mfem::Vector scratch;

  Solver& trPrec;

public:
#ifdef MFEM_USE_MPI
  TrustRegion(MPI_Comm comm_, Solver& tPrec) : mfem::NewtonSolver(comm_), trPrec(tPrec) {}
#endif

  void project_to_boundary_with_coefs(mfem::Vector& z, const mfem::Vector& d, double trSize, double zz, double zd,
                                      double dd) const
  {
    // find tau s.t. (z + tau*d)^2 = trSize^2
    double tau = (std::sqrt((trSize * trSize - zz) * dd + zd * zd) - zd) / dd;
    z.Add(tau, d);
  }

  void project_to_boundary_between_with_coefs(mfem::Vector& z, const mfem::Vector& y, double trSize, double zz,
                                              double zy, double yy) const
  {
    // find tau s.t. (z + tau*(y-z))^2 = trSize^2
    double dd  = yy - 2 * zy + zz;
    double zd  = zy - zz;
    double tau = (std::sqrt((trSize * trSize - zz) * dd + zd * zd) - zd) / dd;
    z.Add(-tau, z);
    z.Add(tau, y);
  }

  double update_step_length_squared(double alpha, double zz, double zd, double dd) const
  {
    return zz + 2 * alpha * zd + alpha * alpha * dd;
  }

  void dogleg_step(const mfem::Vector& cp, const mfem::Vector& newtonP, double trSize, mfem::Vector& s) const
  {
    // MRT, could optimize some of these eventually, compute on the outside and save
    double cc = Dot(cp, cp);
    double nn = Dot(newtonP, newtonP);
    double tt = trSize * trSize;

    s = 0.0;
    if (cc >= tt) {
      add(s, std::sqrt(tt / cc), cp, s);
    } else if (cc > nn) {
      if (print_options.warnings) {
        mfem::out << "cp outside newton, preconditioner likely inaccurate\n";
      }
      add(s, 1.0, cp, s);
    } else if (nn > tt) {  // on the dogleg (we have nn >= cc, and tt >= cc)
      add(s, 1.0, cp, s);
      double cn = Dot(cp, newtonP);
      project_to_boundary_between_with_coefs(s, newtonP, trSize, cc, cn, nn);
    } else {
      s = newtonP;
    }
  }

  template <typename HessVecFunc, typename PrecondFunc>
  void solve_trust_region_minimization(const mfem::Vector& r0, mfem::Vector& rCurrent, HessVecFunc hess_vec_func,
                                       PrecondFunc precond, const TrustRegionSettings& settings, double& trSize,
                                       TrustRegionResults& results) const
  {
    // minimize r@z + 0.5*z@J@z
    results.interiorStatus    = TrustRegionResults::Status::Interior;
    results.cgIterationsCount = 0;

    auto& z      = results.z;
    auto& cgIter = results.cgIterationsCount;
    auto& d      = results.d;
    auto& Pr     = results.Pr;
    auto& Hd     = results.Hd;

    const double cgTolSquared = settings.cgTol * settings.cgTol;

    if (Dot(r0, r0) < cgTolSquared) {
      return;
    }

    rCurrent = r0;
    precond(rCurrent, Pr);
    d = 0.0;
    add(d, -1.0, Pr, d);  // d = -Pr

    z          = 0.0;
    double zz  = 0.;
    double rPr = Dot(rCurrent, Pr);
    double zd  = 0.0;
    double dd  = Dot(d, d);

    for (cgIter = 1; cgIter <= settings.maxCgIterations; ++cgIter) {
      hess_vec_func(d, Hd);
      double curvature = Dot(d, Hd);
      double alpha     = rPr / curvature;

      auto& zPred = Hd;  // re-use Hd, this is where bugs come from
      add(z, alpha, d, zPred);
      double zzNp1 = Dot(zPred, zPred);  // update_step_length_squared(alpha, zz, zd, dd); MRT, perf optimization

      if (curvature <= 0) {
        // mfem::out << "negative curvature found.\n";
        project_to_boundary_with_coefs(z, d, trSize, zz, zd, dd);
        results.interiorStatus = TrustRegionResults::Status::NegativeCurvature;
        return;
      } else if (zzNp1 > (trSize * trSize)) {
        // mfem::out << "step outside trust region.\n";
        project_to_boundary_with_coefs(z, d, trSize, zz, zd, dd);
        results.interiorStatus = TrustRegionResults::Status::OnBoundary;
        return;
      }

      z = zPred;

      hess_vec_func(d, Hd);
      add(rCurrent, alpha, Hd, rCurrent);

      precond(rCurrent, Pr);
      double rPrNp1 = Dot(rCurrent, Pr);

      if (Dot(rCurrent, rCurrent) <= cgTolSquared) {
        // Hd = 0.0;
        // hess_vec_func(z, Hd);
        // double energy = Dot(r0, z) + 0.5 * Dot(z, Hd);
        // std::cout << "converged cg with energy drop = " << energy << std::endl;
        return;
      }

      double beta = rPrNp1 / rPr;
      rPr         = rPrNp1;
      add(-1.0, Pr, beta, d, d);

      zz = zzNp1;
      zd = Dot(z, d);
      dd = Dot(d, d);
    }
  }

  void Mult(const mfem::Vector&, mfem::Vector& x) const
  {
    MFEM_ASSERT(oper != NULL, "the Operator is not set (use SetOperator).");
    MFEM_ASSERT(prec != NULL, "the Solver is not set (use SetSolver).");

    using real_t = mfem::real_t;

    real_t norm, norm_goal;
    oper->Mult(x, r);

    norm = initial_norm = Norm(r);
    norm_goal           = std::max(rel_tol * initial_norm, abs_tol);
    if (print_options.first_and_last && !print_options.iterations) {
      mfem::out << "Newton iteration " << std::setw(2) << 0 << " : ||r|| = " << norm << "...\n";
    }
    prec->iterative_mode  = false;
    trPrec.iterative_mode = false;

    // local arrays
    xPred.SetSize(x.Size());
    xPred = 0.0;
    rPred.SetSize(x.Size());
    rPred = 0.0;
    scratch.SetSize(x.Size());
    scratch = 0.0;

    TrustRegionResults  trResults(x.Size());
    TrustRegionSettings settings;  // MRT, read these in please
    settings.cgTol           = 0.2 * norm_goal;
    double trSize            = 10.0;
    size_t cumulativeCgIters = 0;

    int it = 0;
    for (; true; it++) {
      MFEM_ASSERT(IsFinite(norm), "norm = " << norm);
      if (print_options.iterations) {
        mfem::out << "Newton iteration " << std::setw(2) << it << " : ||r|| = " << norm;
        if (it > 0) {
          mfem::out << ", ||r||/||r_0|| = " << norm / initial_norm;
        }
        mfem::out << '\n';
      }

      if (norm <= norm_goal) {
        converged = true;
        break;
      } else if (it >= max_iter) {
        converged = false;
        break;
      }

      auto K = &oper->GetGradient(x);
      if (it == 0 || (trResults.cgIterationsCount >= settings.maxCgIterations ||
                      cumulativeCgIters >= settings.maxCumulativeIteration)) {
        trPrec.SetOperator(*K);
        cumulativeCgIters = 0;
        if (print_options.iterations) {
          mfem::out << "Updating trust region preconditioner." << std::endl;
        }
      }

      auto hess_vec_func = [=](const mfem::Vector& x, mfem::Vector& v) { K->Mult(x, v); };

      auto precond_func = [=](const mfem::Vector& x, mfem::Vector& v) {
        trPrec.Mult(x, v);
        // v = x;
      };

      double cauchyPointNormSquared = trSize * trSize;
      trResults.reset();

      hess_vec_func(r, trResults.Hd);
      double gKg = Dot(r, trResults.Hd);
      if (gKg > 0) {
        double alpha = -Dot(r, r) / gKg;
        add(trResults.cauchyPoint, alpha, r, trResults.cauchyPoint);
        cauchyPointNormSquared = Dot(trResults.cauchyPoint, trResults.cauchyPoint);
      } else {
        double alpha = -trSize / std::sqrt(Dot(r, r));
        add(trResults.cauchyPoint, alpha, r, trResults.cauchyPoint);
        if (print_options.iterations) {
          mfem::out << "Negative curvature unpreconditioned cauchy point direction found."
                    << "\n";
        }
      }

      if (cauchyPointNormSquared >= trSize * trSize) {
        if (print_options.iterations) {
          mfem::out << "Unpreconditioned gradient cauchy point outside trust region, step size = "
                    << std::sqrt(cauchyPointNormSquared) << "\n";
        }
        trResults.cauchyPoint *= (trSize / std::sqrt(cauchyPointNormSquared));
        trResults.z                 = trResults.cauchyPoint;
        trResults.cgIterationsCount = 1;
        trResults.interiorStatus    = TrustRegionResults::Status::OnBoundary;
      } else {
        solve_trust_region_minimization(r, scratch, hess_vec_func, precond_func, settings, trSize, trResults);
        if (print_options.iterations) {
          mfem::out << "Trust region linear solve took " << trResults.cgIterationsCount << " cg iterations.\n";
        }
      }
      cumulativeCgIters += trResults.cgIterationsCount;

      bool happyAboutTrSize = false;
      while (!happyAboutTrSize) {
        auto& d  = trResults.d;   // reuse, dangerous!
        auto& Hd = trResults.Hd;  // reuse, dangerous!

        dogleg_step(trResults.cauchyPoint, trResults.z, trSize, d);

        hess_vec_func(d, Hd);
        double dHd            = Dot(d, Hd);
        double modelObjective = Dot(r, d) + 0.5 * dHd;

        add(x, d, xPred);
        oper->Mult(xPred, rPred);
        double realObjective = 0.5 * (Dot(r, d) + Dot(rPred, d));

        double normPred = Norm(rPred);
        MFEM_ASSERT(IsFinite(normPred), "norm = " << normPred);

        if (normPred <= norm_goal) {
          x                = xPred;
          r                = rPred;
          norm             = normPred;
          happyAboutTrSize = true;
          break;
        }

        double modelImprove = -modelObjective;
        double realImprove  = -realObjective;

        double rho = realImprove / modelImprove;
        if (modelObjective > 0) {
          if (print_options.iterations || print_options.warnings) {
            mfem::out << "Found a positive model objective increase.  Debug if you see this.\n";
          }
          rho = realImprove / -modelImprove;
        }

        if (!(rho >= settings.eta2)) {  // write it this way to handle NaNs
          trSize *= settings.t1;
        } else if ((rho > settings.eta3) && (trResults.interiorStatus == TrustRegionResults::Status::OnBoundary)) {
          trSize *= settings.t2;
        }

        // eventually extend to handle this case
        // modelRes = g + Jd
        // modelResNorm = np.linalg.norm(modelRes)
        // realResNorm = np.linalg.norm(gy)
        bool willAccept = rho >= settings.eta1;  // or (rho >= -0 and realResNorm <= gNorm)

        if (willAccept) {
          x                = xPred;
          r                = rPred;
          norm             = normPred;
          happyAboutTrSize = true;
          break;
        }
      }
    }

    final_iter = it;
    final_norm = norm;

    if (print_options.summary || (!converged && print_options.warnings) || print_options.first_and_last) {
      mfem::out << "Newton: Number of iterations: " << final_iter << '\n' << "   ||r|| = " << final_norm << '\n';
    }
    if (!converged && (print_options.summary || print_options.warnings)) {
      mfem::out << "Newton: No convergence!\n";
    }
  }
};

bool usePreconditionerInsteadOfLinearSolve(const mfem::NewtonSolver* const nonlinearSolver)
{
  if (dynamic_cast<const TrustRegion*>(nonlinearSolver)) {
    return true;
  }
  return false;
}

EquationSolver::EquationSolver(NonlinearSolverOptions nonlinear_opts, LinearSolverOptions lin_opts, MPI_Comm comm)
{
  auto [lin_solver, preconditioner] = buildLinearSolverAndPreconditioner(lin_opts, comm);

  lin_solver_     = std::move(lin_solver);
  preconditioner_ = std::move(preconditioner);
  nonlin_solver_  = buildNonlinearSolver(nonlinear_opts, *preconditioner_, comm);
}

EquationSolver::EquationSolver(std::unique_ptr<mfem::NewtonSolver> nonlinear_solver,
                               std::unique_ptr<mfem::Solver>       linear_solver,
                               std::unique_ptr<mfem::Solver>       preconditioner)
{
  SLIC_ERROR_ROOT_IF(!nonlinear_solver, "Nonlinear solvers must be given to construct an EquationSolver");
  SLIC_ERROR_ROOT_IF(!linear_solver, "Linear solvers must be given to construct an EquationSolver");

  nonlin_solver_  = std::move(nonlinear_solver);
  lin_solver_     = std::move(linear_solver);
  preconditioner_ = std::move(preconditioner);
}

void EquationSolver::setOperator(const mfem::Operator& op)
{
  nonlin_solver_->SetOperator(op);

  // Now that the nonlinear solver knows about the operator, we can set its linear solver
  if (!nonlin_solver_set_solver_called_) {
    nonlin_solver_->SetSolver(linearSolver());
    nonlin_solver_set_solver_called_ = true;
  }
}

void EquationSolver::solve(mfem::Vector& x) const
{
  mfem::Vector zero(x);
  zero = 0.0;
  // KINSOL does not handle non-zero RHS, so we enforce that the RHS
  // of the nonlinear system is zero
  nonlin_solver_->Mult(zero, x);
}

void SuperLUSolver::Mult(const mfem::Vector& input, mfem::Vector& output) const
{
  SLIC_ERROR_ROOT_IF(!superlu_mat_, "Operator must be set prior to solving with SuperLU");

  // Use the underlying MFEM-based solver and SuperLU matrix type to solve the system
  superlu_solver_.Mult(input, output);
}

/**
 * @brief Function for building a monolithic parallel Hypre matrix from a block system of smaller Hypre matrices
 *
 * @param block_operator The block system of HypreParMatrices
 * @return The assembled monolithic HypreParMatrix
 *
 * @pre @a block_operator must have assembled HypreParMatrices for its sub-blocks
 */
std::unique_ptr<mfem::HypreParMatrix> buildMonolithicMatrix(const mfem::BlockOperator& block_operator)
{
  int row_blocks = block_operator.NumRowBlocks();
  int col_blocks = block_operator.NumColBlocks();

  SLIC_ERROR_ROOT_IF(row_blocks != col_blocks, "Attempted to use a direct solver on a non-square block system.");

  mfem::Array2D<const mfem::HypreParMatrix*> hypre_blocks(row_blocks, col_blocks);

  for (int i = 0; i < row_blocks; ++i) {
    for (int j = 0; j < col_blocks; ++j) {
      // checks for presence of empty (null) blocks, which happen fairly common in multirank contact
      if (!block_operator.IsZeroBlock(i, j)) {
        auto* hypre_block = dynamic_cast<const mfem::HypreParMatrix*>(&block_operator.GetBlock(i, j));
        SLIC_ERROR_ROOT_IF(!hypre_block,
                           "Trying to use SuperLU on a block operator that does not contain HypreParMatrix blocks.");

        hypre_blocks(i, j) = hypre_block;
      } else {
        hypre_blocks(i, j) = nullptr;
      }
    }
  }

  // Note that MFEM passes ownership of this matrix to the caller
  return std::unique_ptr<mfem::HypreParMatrix>(mfem::HypreParMatrixFromBlocks(hypre_blocks));
}

void SuperLUSolver::SetOperator(const mfem::Operator& op)
{
  // Check if this is a block operator
  auto* block_operator = dynamic_cast<const mfem::BlockOperator*>(&op);

  // If it is, make a monolithic system from the underlying blocks
  if (block_operator) {
    auto monolithic_mat = buildMonolithicMatrix(*block_operator);

    superlu_mat_ = std::make_unique<mfem::SuperLURowLocMatrix>(*monolithic_mat);
  } else {
    // If this is not a block system, check that the input operator is a HypreParMatrix as expected
    auto* matrix = dynamic_cast<const mfem::HypreParMatrix*>(&op);

    SLIC_ERROR_ROOT_IF(!matrix, "Matrix must be an assembled HypreParMatrix for use with SuperLU");

    superlu_mat_ = std::make_unique<mfem::SuperLURowLocMatrix>(*matrix);
  }

  superlu_solver_.SetOperator(*superlu_mat_);
}

#ifdef MFEM_USE_STRUMPACK

void StrumpackSolver::Mult(const mfem::Vector& input, mfem::Vector& output) const
{
  SLIC_ERROR_ROOT_IF(!strumpack_mat_, "Operator must be set prior to solving with Strumpack");

  // Use the underlying MFEM-based solver and Strumpack matrix type to solve the system
  strumpack_solver_.Mult(input, output);
}

void StrumpackSolver::SetOperator(const mfem::Operator& op)
{
  // Check if this is a block operator
  auto* block_operator = dynamic_cast<const mfem::BlockOperator*>(&op);

  // If it is, make a monolithic system from the underlying blocks
  if (block_operator) {
    auto monolithic_mat = buildMonolithicMatrix(*block_operator);

    strumpack_mat_ = std::make_unique<mfem::STRUMPACKRowLocMatrix>(*monolithic_mat);
  } else {
    // If this is not a block system, check that the input operator is a HypreParMatrix as expected
    auto* matrix = dynamic_cast<const mfem::HypreParMatrix*>(&op);

    SLIC_ERROR_ROOT_IF(!matrix, "Matrix must be an assembled HypreParMatrix for use with Strumpack");

    strumpack_mat_ = std::make_unique<mfem::STRUMPACKRowLocMatrix>(*matrix);
  }

  strumpack_solver_.SetOperator(*strumpack_mat_);
}

#endif

std::unique_ptr<mfem::NewtonSolver> buildNonlinearSolver(const NonlinearSolverOptions& nonlinear_opts,
                                                         mfem::Solver& prec, MPI_Comm comm)
{
  std::unique_ptr<mfem::NewtonSolver> nonlinear_solver;

  if (nonlinear_opts.nonlin_solver == NonlinearSolver::Newton) {
    nonlinear_solver = std::make_unique<mfem::NewtonSolver>(comm);
  } else if (nonlinear_opts.nonlin_solver == NonlinearSolver::LBFGS) {
    nonlinear_solver = std::make_unique<mfem::LBFGSSolver>(comm);
  } else if (nonlinear_opts.nonlin_solver == NonlinearSolver::NewtonLineSearch) {
    nonlinear_solver = std::make_unique<NewtonSolver>(comm, nonlinear_opts);
  } else if (nonlinear_opts.nonlin_solver == NonlinearSolver::Nesterov) {
    nonlinear_solver = std::make_unique<Nesterov>(comm);
  } else if (nonlinear_opts.nonlin_solver == NonlinearSolver::TrustRegion) {
    nonlinear_solver = std::make_unique<TrustRegion>(comm, prec);
  }
  // KINSOL
  else {
#ifdef SERAC_USE_SUNDIALS

    int kinsol_strat = KIN_NONE;

    switch (nonlinear_opts.nonlin_solver) {
      case NonlinearSolver::KINFullStep:
        kinsol_strat = KIN_NONE;
        break;
      case NonlinearSolver::KINBacktrackingLineSearch:
        kinsol_strat = KIN_LINESEARCH;
        break;
      case NonlinearSolver::KINPicard:
        kinsol_strat = KIN_PICARD;
        break;
      default:
        kinsol_strat = KIN_NONE;
        SLIC_ERROR_ROOT("Unknown KINSOL nonlinear solver type given.");
    }
    auto kinsol_solver = std::make_unique<mfem::KINSolver>(comm, kinsol_strat, true);
    nonlinear_solver   = std::move(kinsol_solver);
#else
    SLIC_ERROR_ROOT("KINSOL was not enabled when MFEM was built");
#endif
  }

  nonlinear_solver->SetRelTol(nonlinear_opts.relative_tol);
  nonlinear_solver->SetAbsTol(nonlinear_opts.absolute_tol);
  nonlinear_solver->SetMaxIter(nonlinear_opts.max_iterations);
  nonlinear_solver->SetPrintLevel(nonlinear_opts.print_level);

  // Iterative mode indicates we do not zero out the initial guess during the
  // nonlinear solver call. This is required as we apply the essential boundary
  // conditions before the nonlinear solver is applied.
  nonlinear_solver->iterative_mode = true;

  return nonlinear_solver;
}

std::pair<std::unique_ptr<mfem::Solver>, std::unique_ptr<mfem::Solver>> buildLinearSolverAndPreconditioner(
    LinearSolverOptions linear_opts, MPI_Comm comm)
{
  auto preconditioner = buildPreconditioner(linear_opts.preconditioner, linear_opts.preconditioner_print_level, comm);

  if (linear_opts.linear_solver == LinearSolver::SuperLU) {
    auto lin_solver = std::make_unique<SuperLUSolver>(linear_opts.print_level, comm);
    return {std::move(lin_solver), std::move(preconditioner)};
  }

#ifdef MFEM_USE_STRUMPACK

  if (linear_opts.linear_solver == LinearSolver::Strumpack) {
    auto lin_solver = std::make_unique<StrumpackSolver>(linear_opts.print_level, comm);
    return {std::move(lin_solver), std::move(preconditioner)};
  }

#endif

  std::unique_ptr<mfem::IterativeSolver> iter_lin_solver;

  switch (linear_opts.linear_solver) {
    case LinearSolver::CG:
      iter_lin_solver = std::make_unique<mfem::CGSolver>(comm);
      break;
    case LinearSolver::GMRES:
      iter_lin_solver = std::make_unique<mfem::GMRESSolver>(comm);
      break;
    default:
      SLIC_ERROR_ROOT("Linear solver type not recognized.");
      exitGracefully(true);
  }

  iter_lin_solver->SetRelTol(linear_opts.relative_tol);
  iter_lin_solver->SetAbsTol(linear_opts.absolute_tol);
  iter_lin_solver->SetMaxIter(linear_opts.max_iterations);
  iter_lin_solver->SetPrintLevel(linear_opts.print_level);

  if (preconditioner) {
    iter_lin_solver->SetPreconditioner(*preconditioner);
  }

  return {std::move(iter_lin_solver), std::move(preconditioner)};
}

#ifdef MFEM_USE_AMGX
std::unique_ptr<mfem::AmgXSolver> buildAMGX(const AMGXOptions& options, const MPI_Comm comm)
{
  auto          amgx = std::make_unique<mfem::AmgXSolver>();
  conduit::Node options_node;
  options_node["config_version"] = 2;
  auto& solver_options           = options_node["solver"];
  solver_options["solver"]       = "AMG";
  solver_options["presweeps"]    = 1;
  solver_options["postsweeps"]   = 2;
  solver_options["interpolator"] = "D2";
  solver_options["max_iters"]    = 2;
  solver_options["convergence"]  = "ABSOLUTE";
  solver_options["cycle"]        = "V";

  if (options.verbose) {
    options_node["solver/obtain_timings"]    = 1;
    options_node["solver/monitor_residual"]  = 1;
    options_node["solver/print_solve_stats"] = 1;
  }

  // TODO: Use magic_enum here when we can switch to GCC 9+
  // This is an immediately-invoked lambda so that the map
  // can be const without needed to initialize all the values
  // in the constructor
  static const auto solver_names = []() {
    std::unordered_map<AMGXSolver, std::string> names;
    names[AMGXSolver::AMG]             = "AMG";
    names[AMGXSolver::PCGF]            = "PCGF";
    names[AMGXSolver::CG]              = "CG";
    names[AMGXSolver::PCG]             = "PCG";
    names[AMGXSolver::PBICGSTAB]       = "PBICGSTAB";
    names[AMGXSolver::BICGSTAB]        = "BICGSTAB";
    names[AMGXSolver::FGMRES]          = "FGMRES";
    names[AMGXSolver::JACOBI_L1]       = "JACOBI_L1";
    names[AMGXSolver::GS]              = "GS";
    names[AMGXSolver::POLYNOMIAL]      = "POLYNOMIAL";
    names[AMGXSolver::KPZ_POLYNOMIAL]  = "KPZ_POLYNOMIAL";
    names[AMGXSolver::BLOCK_JACOBI]    = "BLOCK_JACOBI";
    names[AMGXSolver::MULTICOLOR_GS]   = "MULTICOLOR_GS";
    names[AMGXSolver::MULTICOLOR_DILU] = "MULTICOLOR_DILU";
    return names;
  }();

  options_node["solver/solver"]   = solver_names.at(options.solver);
  options_node["solver/smoother"] = solver_names.at(options.smoother);

  // Treat the string as the config (not a filename)
  amgx->ReadParameters(options_node.to_json(), mfem::AmgXSolver::INTERNAL);
  amgx->InitExclusiveGPU(comm);

  return amgx;
}
#endif

std::unique_ptr<mfem::Solver> buildPreconditioner(Preconditioner preconditioner, int print_level,
                                                  [[maybe_unused]] MPI_Comm comm)
{
  std::unique_ptr<mfem::Solver> preconditioner_solver;

  // Handle the preconditioner - currently just BoomerAMG and HypreSmoother are supported
  if (preconditioner == Preconditioner::HypreAMG) {
    auto amg_preconditioner = std::make_unique<mfem::HypreBoomerAMG>();
    amg_preconditioner->SetPrintLevel(print_level);
    preconditioner_solver = std::move(amg_preconditioner);
  } else if (preconditioner == Preconditioner::HypreJacobi) {
    auto jac_preconditioner = std::make_unique<mfem::HypreSmoother>();
    jac_preconditioner->SetType(mfem::HypreSmoother::Type::Jacobi);
    preconditioner_solver = std::move(jac_preconditioner);
  } else if (preconditioner == Preconditioner::HypreL1Jacobi) {
    auto jacl1_preconditioner = std::make_unique<mfem::HypreSmoother>();
    jacl1_preconditioner->SetType(mfem::HypreSmoother::Type::l1Jacobi);
    preconditioner_solver = std::move(jacl1_preconditioner);
  } else if (preconditioner == Preconditioner::HypreGaussSeidel) {
    auto gs_preconditioner = std::make_unique<mfem::HypreSmoother>();
    gs_preconditioner->SetType(mfem::HypreSmoother::Type::GS);
    preconditioner_solver = std::move(gs_preconditioner);
  } else if (preconditioner == Preconditioner::HypreILU) {
    auto ilu_preconditioner = std::make_unique<mfem::HypreILU>();
    ilu_preconditioner->SetLevelOfFill(1);
    ilu_preconditioner->SetPrintLevel(print_level);
    preconditioner_solver = std::move(ilu_preconditioner);
  } else if (preconditioner == Preconditioner::AMGX) {
#ifdef MFEM_USE_AMGX
    preconditioner_solver = buildAMGX(AMGXOptions{}, comm);
#else
    SLIC_ERROR_ROOT("AMGX requested in non-GPU build");
#endif
  } else {
    SLIC_ERROR_ROOT_IF(preconditioner != Preconditioner::None, "Unknown preconditioner type requested");
  }

  return preconditioner_solver;
}

void EquationSolver::defineInputFileSchema(axom::inlet::Container& container)
{
  auto& linear_container = container.addStruct("linear", "Linear Equation Solver Parameters");
  linear_container.required().registerVerifier([](const axom::inlet::Container& container_to_verify) {
    // Make sure that the provided options match the desired linear solver type
    const bool is_iterative = (container_to_verify["type"].get<std::string>() == "iterative") &&
                              container_to_verify.contains("iterative_options");
    const bool is_direct =
        (container_to_verify["type"].get<std::string>() == "direct") && container_to_verify.contains("direct_options");
    return is_iterative || is_direct;
  });

  // Enforce the solver type - must be iterative or direct
  linear_container.addString("type", "The type of solver parameters to use (iterative|direct)")
      .required()
      .validValues({"iterative", "direct"});

  auto& iterative_container = linear_container.addStruct("iterative_options", "Iterative solver parameters");
  iterative_container.addDouble("rel_tol", "Relative tolerance for the linear solve.").defaultValue(1.0e-6);
  iterative_container.addDouble("abs_tol", "Absolute tolerance for the linear solve.").defaultValue(1.0e-8);
  iterative_container.addInt("max_iter", "Maximum iterations for the linear solve.").defaultValue(5000);
  iterative_container.addInt("print_level", "Linear print level.").defaultValue(0);
  iterative_container.addString("solver_type", "Solver type (gmres|minres|cg).").defaultValue("gmres");
  iterative_container.addString("prec_type", "Preconditioner type (JacobiSmoother|L1JacobiSmoother|AMG|ILU).")
      .defaultValue("JacobiSmoother");

  auto& direct_container = linear_container.addStruct("direct_options", "Direct solver parameters");
  direct_container.addInt("print_level", "Linear print level.").defaultValue(0);

  // Only needed for nonlinear problems
  auto& nonlinear_container = container.addStruct("nonlinear", "Newton Equation Solver Parameters").required(false);
  nonlinear_container.addDouble("rel_tol", "Relative tolerance for the Newton solve.").defaultValue(1.0e-2);
  nonlinear_container.addDouble("abs_tol", "Absolute tolerance for the Newton solve.").defaultValue(1.0e-4);
  nonlinear_container.addInt("max_iter", "Maximum iterations for the Newton solve.").defaultValue(500);
  nonlinear_container.addInt("print_level", "Nonlinear print level.").defaultValue(0);
  nonlinear_container.addString("solver_type", "Solver type (Newton|KINFullStep|KINLineSearch)").defaultValue("Newton");
}

}  // namespace serac

using serac::EquationSolver;
using serac::LinearSolverOptions;
using serac::NonlinearSolverOptions;

serac::LinearSolverOptions FromInlet<serac::LinearSolverOptions>::operator()(const axom::inlet::Container& base)
{
  LinearSolverOptions options;
  std::string         type = base["type"];

  if (type == "direct") {
    options.linear_solver = serac::LinearSolver::SuperLU;
    options.print_level   = base["direct_options/print_level"];
    return options;
  }

  auto config             = base["iterative_options"];
  options.relative_tol    = config["rel_tol"];
  options.absolute_tol    = config["abs_tol"];
  options.max_iterations  = config["max_iter"];
  options.print_level     = config["print_level"];
  std::string solver_type = config["solver_type"];
  if (solver_type == "gmres") {
    options.linear_solver = serac::LinearSolver::GMRES;
  } else if (solver_type == "cg") {
    options.linear_solver = serac::LinearSolver::CG;
  } else {
    std::string msg = axom::fmt::format("Unknown Linear solver type given: '{0}'", solver_type);
    SLIC_ERROR_ROOT(msg);
  }
  const std::string prec_type = config["prec_type"];
  if (prec_type == "JacobiSmoother") {
    options.preconditioner = serac::Preconditioner::HypreJacobi;
  } else if (prec_type == "L1JacobiSmoother") {
    options.preconditioner = serac::Preconditioner::HypreL1Jacobi;
  } else if (prec_type == "HypreAMG") {
    options.preconditioner = serac::Preconditioner::HypreAMG;
  } else if (prec_type == "ILU") {
    options.preconditioner = serac::Preconditioner::HypreILU;
#ifdef MFEM_USE_AMGX
  } else if (prec_type == "AMGX") {
    options.preconditioner = serac::Preconditioner::AMGX;
#endif
  } else if (prec_type == "GaussSeidel") {
    options.preconditioner = serac::Preconditioner::HypreGaussSeidel;
  } else {
    std::string msg = axom::fmt::format("Unknown preconditioner type given: '{0}'", prec_type);
    SLIC_ERROR_ROOT(msg);
  }

  return options;
}

serac::NonlinearSolverOptions FromInlet<serac::NonlinearSolverOptions>::operator()(const axom::inlet::Container& base)
{
  NonlinearSolverOptions options;
  options.relative_tol          = base["rel_tol"];
  options.absolute_tol          = base["abs_tol"];
  options.max_iterations        = base["max_iter"];
  options.print_level           = base["print_level"];
  const std::string solver_type = base["solver_type"];
  if (solver_type == "Newton") {
    options.nonlin_solver = serac::NonlinearSolver::Newton;
  } else if (solver_type == "KINFullStep") {
    options.nonlin_solver = serac::NonlinearSolver::KINFullStep;
  } else if (solver_type == "KINLineSearch") {
    options.nonlin_solver = serac::NonlinearSolver::KINBacktrackingLineSearch;
  } else if (solver_type == "KINPicard") {
    options.nonlin_solver = serac::NonlinearSolver::KINPicard;
  } else {
    SLIC_ERROR_ROOT(axom::fmt::format("Unknown nonlinear solver type given: '{0}'", solver_type));
  }
  return options;
}

serac::EquationSolver FromInlet<serac::EquationSolver>::operator()(const axom::inlet::Container& base)
{
  auto lin    = base["linear"].get<LinearSolverOptions>();
  auto nonlin = base["nonlinear"].get<NonlinearSolverOptions>();

  auto [linear_solver, preconditioner] = serac::buildLinearSolverAndPreconditioner(lin, MPI_COMM_WORLD);

  serac::EquationSolver eq_solver(serac::buildNonlinearSolver(nonlin, *preconditioner, MPI_COMM_WORLD),
                                  std::move(linear_solver), std::move(preconditioner));

  return eq_solver;
}
