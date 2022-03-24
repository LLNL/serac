#include "mfem.hpp"
#include "mfem/general/forall.hpp"

#include "axom/core/utilities/Timer.hpp"

#include "serac/infrastructure/accelerator.hpp"

#include "serac/numerics/functional/tensor.hpp"
#include "serac/numerics/functional/quadrature.hpp"
#include "serac/numerics/functional/finite_element.hpp"
#include "serac/numerics/functional/tuple_arithmetic.hpp"
#include "serac/numerics/functional/integral_utilities.hpp"

#include "sum_factorization.hpp"
#include "sum_factorization_external_cache.hpp"
namespace mfem {

template <int T_D1D = 0, int T_Q1D = 0>
static void PAMassApply3D(const int NE, const Array<double>& b_, const Array<double>& bt_, const Vector& d_,
                          const Vector& x_, Vector& y_, const int d1d = 0, const int q1d = 0)
{
  const int D1D = T_D1D ? T_D1D : d1d;
  const int Q1D = T_Q1D ? T_Q1D : q1d;
  MFEM_VERIFY(D1D <= MAX_D1D, "");
  MFEM_VERIFY(Q1D <= MAX_Q1D, "");
  auto B  = Reshape(b_.Read(), Q1D, D1D);
  auto Bt = Reshape(bt_.Read(), D1D, Q1D);
  auto D  = Reshape(d_.Read(), Q1D, Q1D, Q1D, NE);
  auto X  = Reshape(x_.Read(), D1D, D1D, D1D, NE);
  auto Y  = Reshape(y_.ReadWrite(), D1D, D1D, D1D, NE);
  MFEM_FORALL(e, NE, {
    const int     D1D     = T_D1D ? T_D1D : d1d;
    const int     Q1D     = T_Q1D ? T_Q1D : q1d;
    constexpr int max_D1D = T_D1D ? T_D1D : MAX_D1D;
    constexpr int max_Q1D = T_Q1D ? T_Q1D : MAX_Q1D;
    double        sol_xyz[max_Q1D][max_Q1D][max_Q1D];
    for (int qz = 0; qz < Q1D; ++qz) {
      for (int qy = 0; qy < Q1D; ++qy) {
        for (int qx = 0; qx < Q1D; ++qx) {
          sol_xyz[qz][qy][qx] = 0.0;
        }
      }
    }
    for (int dz = 0; dz < D1D; ++dz) {
      double sol_xy[max_Q1D][max_Q1D];
      for (int qy = 0; qy < Q1D; ++qy) {
        for (int qx = 0; qx < Q1D; ++qx) {
          sol_xy[qy][qx] = 0.0;
        }
      }
      for (int dy = 0; dy < D1D; ++dy) {
        double sol_x[max_Q1D];
        for (int qx = 0; qx < Q1D; ++qx) {
          sol_x[qx] = 0;
        }
        for (int dx = 0; dx < D1D; ++dx) {
          const double s = X(dx, dy, dz, e);
          for (int qx = 0; qx < Q1D; ++qx) {
            sol_x[qx] += B(qx, dx) * s;
          }
        }
        for (int qy = 0; qy < Q1D; ++qy) {
          const double wy = B(qy, dy);
          for (int qx = 0; qx < Q1D; ++qx) {
            sol_xy[qy][qx] += wy * sol_x[qx];
          }
        }
      }
      for (int qz = 0; qz < Q1D; ++qz) {
        const double wz = B(qz, dz);
        for (int qy = 0; qy < Q1D; ++qy) {
          for (int qx = 0; qx < Q1D; ++qx) {
            sol_xyz[qz][qy][qx] += wz * sol_xy[qy][qx];
          }
        }
      }
    }
    for (int qz = 0; qz < Q1D; ++qz) {
      for (int qy = 0; qy < Q1D; ++qy) {
        for (int qx = 0; qx < Q1D; ++qx) {
          sol_xyz[qz][qy][qx] *= D(qx, qy, qz, e);
        }
      }
    }
    for (int qz = 0; qz < Q1D; ++qz) {
      double sol_xy[max_D1D][max_D1D];
      for (int dy = 0; dy < D1D; ++dy) {
        for (int dx = 0; dx < D1D; ++dx) {
          sol_xy[dy][dx] = 0;
        }
      }
      for (int qy = 0; qy < Q1D; ++qy) {
        double sol_x[max_D1D];
        for (int dx = 0; dx < D1D; ++dx) {
          sol_x[dx] = 0;
        }
        for (int qx = 0; qx < Q1D; ++qx) {
          const double s = sol_xyz[qz][qy][qx];
          for (int dx = 0; dx < D1D; ++dx) {
            sol_x[dx] += Bt(dx, qx) * s;
          }
        }
        for (int dy = 0; dy < D1D; ++dy) {
          const double wy = Bt(dy, qy);
          for (int dx = 0; dx < D1D; ++dx) {
            sol_xy[dy][dx] += wy * sol_x[dx];
          }
        }
      }
      for (int dz = 0; dz < D1D; ++dz) {
        const double wz = Bt(dz, qz);
        for (int dy = 0; dy < D1D; ++dy) {
          for (int dx = 0; dx < D1D; ++dx) {
            Y(dx, dy, dz, e) += wz * sol_xy[dy][dx];
          }
        }
      }
    }
  });
}

template <int T_D1D = 0, int T_Q1D = 0>
static void PADiffusionApply3D(const int NE, const bool symmetric, const Array<double>& b, const Array<double>& g,
                               const Array<double>& bt, const Array<double>& gt, const Vector& d_, const Vector& x_,
                               Vector& y_, int d1d = 0, int q1d = 0)
{
  const int D1D = T_D1D ? T_D1D : d1d;
  const int Q1D = T_Q1D ? T_Q1D : q1d;
  MFEM_VERIFY(D1D <= MAX_D1D, "");
  MFEM_VERIFY(Q1D <= MAX_Q1D, "");
  auto B  = Reshape(b.Read(), Q1D, D1D);
  auto G  = Reshape(g.Read(), Q1D, D1D);
  auto Bt = Reshape(bt.Read(), D1D, Q1D);
  auto Gt = Reshape(gt.Read(), D1D, Q1D);
  auto D  = Reshape(d_.Read(), Q1D * Q1D * Q1D, symmetric ? 6 : 9, NE);
  auto X  = Reshape(x_.Read(), D1D, D1D, D1D, NE);
  auto Y  = Reshape(y_.ReadWrite(), D1D, D1D, D1D, NE);
  MFEM_FORALL(e, NE, {
    const int     D1D     = T_D1D ? T_D1D : d1d;
    const int     Q1D     = T_Q1D ? T_Q1D : q1d;
    constexpr int max_D1D = T_D1D ? T_D1D : MAX_D1D;
    constexpr int max_Q1D = T_Q1D ? T_Q1D : MAX_Q1D;
    double        grad[max_Q1D][max_Q1D][max_Q1D][3];
    for (int qz = 0; qz < Q1D; ++qz) {
      for (int qy = 0; qy < Q1D; ++qy) {
        for (int qx = 0; qx < Q1D; ++qx) {
          grad[qz][qy][qx][0] = 0.0;
          grad[qz][qy][qx][1] = 0.0;
          grad[qz][qy][qx][2] = 0.0;
        }
      }
    }
    for (int dz = 0; dz < D1D; ++dz) {
      double gradXY[max_Q1D][max_Q1D][3];
      for (int qy = 0; qy < Q1D; ++qy) {
        for (int qx = 0; qx < Q1D; ++qx) {
          gradXY[qy][qx][0] = 0.0;
          gradXY[qy][qx][1] = 0.0;
          gradXY[qy][qx][2] = 0.0;
        }
      }
      for (int dy = 0; dy < D1D; ++dy) {
        double gradX[max_Q1D][2];
        for (int qx = 0; qx < Q1D; ++qx) {
          gradX[qx][0] = 0.0;
          gradX[qx][1] = 0.0;
        }
        for (int dx = 0; dx < D1D; ++dx) {
          const double s = X(dx, dy, dz, e);
          for (int qx = 0; qx < Q1D; ++qx) {
            gradX[qx][0] += s * B(qx, dx);
            gradX[qx][1] += s * G(qx, dx);
          }
        }
        for (int qy = 0; qy < Q1D; ++qy) {
          const double wy  = B(qy, dy);
          const double wDy = G(qy, dy);
          for (int qx = 0; qx < Q1D; ++qx) {
            const double wx  = gradX[qx][0];
            const double wDx = gradX[qx][1];
            gradXY[qy][qx][0] += wDx * wy;
            gradXY[qy][qx][1] += wx * wDy;
            gradXY[qy][qx][2] += wx * wy;
          }
        }
      }
      for (int qz = 0; qz < Q1D; ++qz) {
        const double wz  = B(qz, dz);
        const double wDz = G(qz, dz);
        for (int qy = 0; qy < Q1D; ++qy) {
          for (int qx = 0; qx < Q1D; ++qx) {
            grad[qz][qy][qx][0] += gradXY[qy][qx][0] * wz;
            grad[qz][qy][qx][1] += gradXY[qy][qx][1] * wz;
            grad[qz][qy][qx][2] += gradXY[qy][qx][2] * wDz;
          }
        }
      }
    }
    // Calculate Dxyz, xDyz, xyDz in plane
    for (int qz = 0; qz < Q1D; ++qz) {
      for (int qy = 0; qy < Q1D; ++qy) {
        for (int qx = 0; qx < Q1D; ++qx) {
          const int    q      = qx + (qy + qz * Q1D) * Q1D;
          const double O11    = D(q, 0, e);
          const double O12    = D(q, 1, e);
          const double O13    = D(q, 2, e);
          const double O21    = symmetric ? O12 : D(q, 3, e);
          const double O22    = symmetric ? D(q, 3, e) : D(q, 4, e);
          const double O23    = symmetric ? D(q, 4, e) : D(q, 5, e);
          const double O31    = symmetric ? O13 : D(q, 6, e);
          const double O32    = symmetric ? O23 : D(q, 7, e);
          const double O33    = symmetric ? D(q, 5, e) : D(q, 8, e);
          const double gradX  = grad[qz][qy][qx][0];
          const double gradY  = grad[qz][qy][qx][1];
          const double gradZ  = grad[qz][qy][qx][2];
          grad[qz][qy][qx][0] = (O11 * gradX) + (O12 * gradY) + (O13 * gradZ);
          grad[qz][qy][qx][1] = (O21 * gradX) + (O22 * gradY) + (O23 * gradZ);
          grad[qz][qy][qx][2] = (O31 * gradX) + (O32 * gradY) + (O33 * gradZ);
        }
      }
    }
    for (int qz = 0; qz < Q1D; ++qz) {
      double gradXY[max_D1D][max_D1D][3];
      for (int dy = 0; dy < D1D; ++dy) {
        for (int dx = 0; dx < D1D; ++dx) {
          gradXY[dy][dx][0] = 0;
          gradXY[dy][dx][1] = 0;
          gradXY[dy][dx][2] = 0;
        }
      }
      for (int qy = 0; qy < Q1D; ++qy) {
        double gradX[max_D1D][3];
        for (int dx = 0; dx < D1D; ++dx) {
          gradX[dx][0] = 0;
          gradX[dx][1] = 0;
          gradX[dx][2] = 0;
        }
        for (int qx = 0; qx < Q1D; ++qx) {
          const double gX = grad[qz][qy][qx][0];
          const double gY = grad[qz][qy][qx][1];
          const double gZ = grad[qz][qy][qx][2];
          for (int dx = 0; dx < D1D; ++dx) {
            const double wx  = Bt(dx, qx);
            const double wDx = Gt(dx, qx);
            gradX[dx][0] += gX * wDx;
            gradX[dx][1] += gY * wx;
            gradX[dx][2] += gZ * wx;
          }
        }
        for (int dy = 0; dy < D1D; ++dy) {
          const double wy  = Bt(dy, qy);
          const double wDy = Gt(dy, qy);
            for (int dx = 0; dx < D1D; ++dx)
               {
                  gradXY[dy][dx][0] += gradX[dx][0] * wy;
                  gradXY[dy][dx][1] += gradX[dx][1] * wDy;
                  gradXY[dy][dx][2] += gradX[dx][2] * wy;
               }
            }
         }
         for (int dz = 0; dz < D1D; ++dz)
         {
            const double wz  = Bt(dz,qz);
            const double wDz = Gt(dz,qz);
            for (int dy = 0; dy < D1D; ++dy)
            {
               for (int dx = 0; dx < D1D; ++dx)
               {
                  Y(dx,dy,dz,e) +=
                     ((gradXY[dy][dx][0] * wz) +
                      (gradXY[dy][dx][1] * wz) +
                      (gradXY[dy][dx][2] * wDz));
               }
            }
         }
      }
   });
}

template<int T_D1D = 0, int T_Q1D = 0>
static void SmemPAMassApply3D(const int NE,
                              const Array<double> &b_,
                              const Array<double> &bt_,
                              const Vector &d_,
                              const Vector &x_,
                              Vector &y_,
                              const int d1d = 0,
                              const int q1d = 0)
{
   MFEM_CONTRACT_VAR(bt_);
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int M1Q = T_Q1D ? T_Q1D : MAX_Q1D;
   constexpr int M1D = T_D1D ? T_D1D : MAX_D1D;
   MFEM_VERIFY(D1D <= M1D, "");
   MFEM_VERIFY(Q1D <= M1Q, "");
   auto b = Reshape(b_.Read(), Q1D, D1D);
   auto d = Reshape(d_.Read(), Q1D, Q1D, Q1D, NE);
   auto x = Reshape(x_.Read(), D1D, D1D, D1D, NE);
   auto y = Reshape(y_.ReadWrite(), D1D, D1D, D1D, NE);
   MFEM_FORALL_3D(e, NE, Q1D, Q1D, 1,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
      constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;
      MFEM_SHARED double sDQ[MQ1*MD1];
      double (*B)[MD1] = (double (*)[MD1]) sDQ;
      double (*Bt)[MQ1] = (double (*)[MQ1]) sDQ;
      MFEM_SHARED double sm0[MDQ*MDQ*MDQ];
      MFEM_SHARED double sm1[MDQ*MDQ*MDQ];
      double (*X)[MD1][MD1]   = (double (*)[MD1][MD1]) sm0;
      double (*DDQ)[MD1][MQ1] = (double (*)[MD1][MQ1]) sm1;
      double (*DQQ)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) sm0;
      double (*QQQ)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) sm1;
      double (*QQD)[MQ1][MD1] = (double (*)[MQ1][MD1]) sm0;
      double (*QDD)[MD1][MD1] = (double (*)[MD1][MD1]) sm1;
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            MFEM_UNROLL(MD1)
            for (int dz = 0; dz < D1D; ++dz)
            {
               X[dz][dy][dx] = x(dx,dy,dz,e);
            }
         }
         MFEM_FOREACH_THREAD(dx,x,Q1D)
         {
            B[dx][dy] = b(dx,dy);
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double u[D1D];
            MFEM_UNROLL(MD1)
            for (int dz = 0; dz < D1D; dz++)
            {
               u[dz] = 0;
            }
            MFEM_UNROLL(MD1)
            for (int dx = 0; dx < D1D; ++dx)
            {
               MFEM_UNROLL(MD1)
               for (int dz = 0; dz < D1D; ++dz)
               {
                  u[dz] += X[dz][dy][dx] * B[qx][dx];
               }
            }
            MFEM_UNROLL(MD1)
            for (int dz = 0; dz < D1D; ++dz)
            {
               DDQ[dz][dy][qx] = u[dz];
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double u[D1D];
            MFEM_UNROLL(MD1)
            for (int dz = 0; dz < D1D; dz++)
            {
               u[dz] = 0;
            }
            MFEM_UNROLL(MD1)
            for (int dy = 0; dy < D1D; ++dy)
            {
               MFEM_UNROLL(MD1)
               for (int dz = 0; dz < D1D; dz++)
               {
                  u[dz] += DDQ[dz][dy][qx] * B[qy][dy];
               }
            }
            MFEM_UNROLL(MD1)
            for (int dz = 0; dz < D1D; dz++)
            {
               DQQ[dz][qy][qx] = u[dz];
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double u[Q1D];
            MFEM_UNROLL(MQ1)
            for (int qz = 0; qz < Q1D; qz++)
            {
               u[qz] = 0;
            }
            MFEM_UNROLL(MD1)
            for (int dz = 0; dz < D1D; ++dz)
            {
               MFEM_UNROLL(MQ1)
               for (int qz = 0; qz < Q1D; qz++)
               {
                  u[qz] += DQQ[dz][qy][qx] * B[qz][dz];
               }
            }
            MFEM_UNROLL(MQ1)
            for (int qz = 0; qz < Q1D; qz++)
            {
               QQQ[qz][qy][qx] = u[qz] * d(qx,qy,qz,e);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(d,y,D1D)
      {
         MFEM_FOREACH_THREAD(q,x,Q1D)
         {
            Bt[d][q] = b(q,d);
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            double u[Q1D];
            MFEM_UNROLL(MQ1)
            for (int qz = 0; qz < Q1D; ++qz)
            {
               u[qz] = 0;
            }
            MFEM_UNROLL(MQ1)
            for (int qx = 0; qx < Q1D; ++qx)
            {
               MFEM_UNROLL(MQ1)
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  u[qz] += QQQ[qz][qy][qx] * Bt[dx][qx];
               }
            }
            MFEM_UNROLL(MQ1)
            for (int qz = 0; qz < Q1D; ++qz)
            {
               QQD[qz][qy][dx] = u[qz];
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            double u[Q1D];
            MFEM_UNROLL(MQ1)
            for (int qz = 0; qz < Q1D; ++qz)
            {
               u[qz] = 0;
            }
            MFEM_UNROLL(MQ1)
            for (int qy = 0; qy < Q1D; ++qy)
            {
               MFEM_UNROLL(MQ1)
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  u[qz] += QQD[qz][qy][dx] * Bt[dy][qy];
               }
            }
            MFEM_UNROLL(MQ1)
            for (int qz = 0; qz < Q1D; ++qz)
            {
               QDD[qz][dy][dx] = u[qz];
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            double u[D1D];
            MFEM_UNROLL(MD1)
            for (int dz = 0; dz < D1D; ++dz)
            {
               u[dz] = 0;
            }
            MFEM_UNROLL(MQ1)
            for (int qz = 0; qz < Q1D; ++qz)
            {
               MFEM_UNROLL(MD1)
               for (int dz = 0; dz < D1D; ++dz)
               {
                  u[dz] += QDD[qz][dy][dx] * Bt[dz][qz];
               }
            }
            MFEM_UNROLL(MD1)
            for (int dz = 0; dz < D1D; ++dz)
            {
               y(dx,dy,dz,e) += u[dz];
            }
         }
      }
   });
}

template<int T_D1D = 0, int T_Q1D = 0>
static void SmemPADiffusionApply3D(const int NE,
                                   const bool symmetric,
                                   const Array<double> &b_,
                                   const Array<double> &g_,
                                   const Vector &d_,
                                   const Vector &x_,
                                   Vector &y_,
                                   const int d1d = 0,
                                   const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int M1Q = T_Q1D ? T_Q1D : MAX_Q1D;
   constexpr int M1D = T_D1D ? T_D1D : MAX_D1D;
   MFEM_VERIFY(D1D <= M1D, "");
   MFEM_VERIFY(Q1D <= M1Q, "");
   auto b = Reshape(b_.Read(), Q1D, D1D);
   auto g = Reshape(g_.Read(), Q1D, D1D);
   auto d = Reshape(d_.Read(), Q1D, Q1D, Q1D, symmetric ? 6 : 9, NE);
   auto x = Reshape(x_.Read(), D1D, D1D, D1D, NE);
   auto y = Reshape(y_.ReadWrite(), D1D, D1D, D1D, NE);
   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
      constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;
      MFEM_SHARED double sBG[2][MQ1*MD1];
      double (*B)[MD1] = (double (*)[MD1]) (sBG+0);
      double (*G)[MD1] = (double (*)[MD1]) (sBG+1);
      double (*Bt)[MQ1] = (double (*)[MQ1]) (sBG+0);
      double (*Gt)[MQ1] = (double (*)[MQ1]) (sBG+1);
      MFEM_SHARED double sm0[3][MDQ*MDQ*MDQ];
      MFEM_SHARED double sm1[3][MDQ*MDQ*MDQ];
      double (*X)[MD1][MD1]    = (double (*)[MD1][MD1]) (sm0+2);
      double (*DDQ0)[MD1][MQ1] = (double (*)[MD1][MQ1]) (sm0+0);
      double (*DDQ1)[MD1][MQ1] = (double (*)[MD1][MQ1]) (sm0+1);
      double (*DQQ0)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm1+0);
      double (*DQQ1)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm1+1);
      double (*DQQ2)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm1+2);
      double (*QQQ0)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm0+0);
      double (*QQQ1)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm0+1);
      double (*QQQ2)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm0+2);
      double (*QQD0)[MQ1][MD1] = (double (*)[MQ1][MD1]) (sm1+0);
      double (*QQD1)[MQ1][MD1] = (double (*)[MQ1][MD1]) (sm1+1);
      double (*QQD2)[MQ1][MD1] = (double (*)[MQ1][MD1]) (sm1+2);
      double (*QDD0)[MD1][MD1] = (double (*)[MD1][MD1]) (sm0+0);
      double (*QDD1)[MD1][MD1] = (double (*)[MD1][MD1]) (sm0+1);
      double (*QDD2)[MD1][MD1] = (double (*)[MD1][MD1]) (sm0+2);
      MFEM_FOREACH_THREAD(dz,z,D1D)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(dx,x,D1D)
            {
               X[dz][dy][dx] = x(dx,dy,dz,e);
            }
         }
      }
      if (MFEM_THREAD_ID(z) == 0)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               B[qx][dy] = b(qx,dy);
               G[qx][dy] = g(qx,dy);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dz,z,D1D)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double u = 0.0, v = 0.0;
               MFEM_UNROLL(MD1)
               for (int dx = 0; dx < D1D; ++dx)
               {
                  const double coords = X[dz][dy][dx];
                  u += coords * B[qx][dx];
                  v += coords * G[qx][dx];
               }
               DDQ0[dz][dy][qx] = u;
               DDQ1[dz][dy][qx] = v;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dz,z,D1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double u = 0.0, v = 0.0, w = 0.0;
               MFEM_UNROLL(MD1)
               for (int dy = 0; dy < D1D; ++dy)
               {
                  u += DDQ1[dz][dy][qx] * B[qy][dy];
                  v += DDQ0[dz][dy][qx] * G[qy][dy];
                  w += DDQ0[dz][dy][qx] * B[qy][dy];
               }
               DQQ0[dz][qy][qx] = u;
               DQQ1[dz][qy][qx] = v;
               DQQ2[dz][qy][qx] = w;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double u = 0.0, v = 0.0, w = 0.0;
               MFEM_UNROLL(MD1)
               for (int dz = 0; dz < D1D; ++dz)
               {
                  u += DQQ0[dz][qy][qx] * B[qz][dz];
                  v += DQQ1[dz][qy][qx] * B[qz][dz];
                  w += DQQ2[dz][qy][qx] * G[qz][dz];
               }
               const double O11 = d(qx,qy,qz,0,e);
               const double O12 = d(qx,qy,qz,1,e);
               const double O13 = d(qx,qy,qz,2,e);
               const double O21 = symmetric ? O12 : d(qx,qy,qz,3,e);
               const double O22 = symmetric ? d(qx,qy,qz,3,e) : d(qx,qy,qz,4,e);
               const double O23 = symmetric ? d(qx,qy,qz,4,e) : d(qx,qy,qz,5,e);
               const double O31 = symmetric ? O13 : d(qx,qy,qz,6,e);
               const double O32 = symmetric ? O23 : d(qx,qy,qz,7,e);
               const double O33 = symmetric ? d(qx,qy,qz,5,e) : d(qx,qy,qz,8,e);
               const double gX = u;
               const double gY = v;
               const double gZ = w;
               QQQ0[qz][qy][qx] = (O11*gX) + (O12*gY) + (O13*gZ);
               QQQ1[qz][qy][qx] = (O21*gX) + (O22*gY) + (O23*gZ);
               QQQ2[qz][qy][qx] = (O31*gX) + (O32*gY) + (O33*gZ);
            }
         }
      }
      MFEM_SYNC_THREAD;
      if (MFEM_THREAD_ID(z) == 0)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               Bt[dy][qx] = b(qx,dy);
               Gt[dy][qx] = g(qx,dy);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(dx,x,D1D)
            {
               double u = 0.0, v = 0.0, w = 0.0;
               MFEM_UNROLL(MQ1)
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  u += QQQ0[qz][qy][qx] * Gt[dx][qx];
                  v += QQQ1[qz][qy][qx] * Bt[dx][qx];
                  w += QQQ2[qz][qy][qx] * Bt[dx][qx];
               }
               QQD0[qz][qy][dx] = u;
               QQD1[qz][qy][dx] = v;
               QQD2[qz][qy][dx] = w;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(dx,x,D1D)
            {
               double u = 0.0, v = 0.0, w = 0.0;
               MFEM_UNROLL(Q1D)
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  u += QQD0[qz][qy][dx] * Bt[dy][qy];
                  v += QQD1[qz][qy][dx] * Gt[dy][qy];
                  w += QQD2[qz][qy][dx] * Bt[dy][qy];
               }
               QDD0[qz][dy][dx] = u;
               QDD1[qz][dy][dx] = v;
               QDD2[qz][dy][dx] = w;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dz,z,D1D)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(dx,x,D1D)
            {
               double u = 0.0, v = 0.0, w = 0.0;
               MFEM_UNROLL(MQ1)
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  u += QDD0[qz][dy][dx] * Bt[dz][qz];
                  v += QDD1[qz][dy][dx] * Bt[dz][qz];
                  w += QDD2[qz][dy][dx] * Gt[dz][qz];
               }
               y(dx,dy,dz,e) += (u + v + w);
            }
         }
      }
   });
}

}

namespace serac {

template < typename trial_space, Geometry geom, int q >
auto BatchPreprocess(const mfem::DeviceTensor< 4, const double > & u_e, GaussLegendreRule<geom, q> rule, int e) {
  static constexpr int n = trial_space::order + 1;

  if constexpr (geom == Geometry::Hexahedron) {

    tensor< double, q, n > B{};
    tensor< double, q, n > G{};
    for (int i = 0; i < q; i++) {
      B[i] = GaussLobattoInterpolation<n>(rule.points_1D[i]);
      G[i] = GaussLobattoInterpolationDerivative<n>(rule.points_1D[i]);
    }
    auto BT = transpose(B);
    auto GT = transpose(G);

    tensor< value_and_gradient< double, tensor< double, 3 > >, q, q, q> u_q{};

    for (int iz = 0; iz < n; ++iz) {
      tensor< value_and_gradient< double, tensor< double, 2 > >, q, q> interpolated_in_XY{};
      for (int iy = 0; iy < n; ++iy) {
        tensor< value_and_gradient< double, double >, q> interpolated_in_X{};
        for (int ix = 0; ix < n; ++ix) {
          const double s = u_e(ix, iy, iz, e);
          for (int qx = 0; qx < q; ++qx) {
            interpolated_in_X[qx].value += s * BT(ix, qx);
            interpolated_in_X[qx].gradient += s * GT(ix, qx);
          }
        }
        for (int qy = 0; qy < q; ++qy) {
          const double interpolate_in_Y = BT(iy, qy);
          const double differentiate_in_Y = GT(iy, qy);
          for (int qx = 0; qx < q; ++qx) {
            interpolated_in_XY[qy][qx].value       += interpolated_in_X[qx].value    * interpolate_in_Y;
            interpolated_in_XY[qy][qx].gradient[0] += interpolated_in_X[qx].gradient * interpolate_in_Y;
            interpolated_in_XY[qy][qx].gradient[1] += interpolated_in_X[qx].value    * differentiate_in_Y;
          }
        }
      }
      for (int qz = 0; qz < q; ++qz) {
        const double interpolate_in_Z = BT(iz, qz);
        const double differentiate_in_Z = GT(iz, qz);
        for (int qy = 0; qy < q; ++qy) {
          for (int qx = 0; qx < q; ++qx) {
            u_q[qz][qy][qx].value       += interpolated_in_XY[qy][qx].value       * interpolate_in_Z;
            u_q[qz][qy][qx].gradient[0] += interpolated_in_XY[qy][qx].gradient[0] * interpolate_in_Z;
            u_q[qz][qy][qx].gradient[1] += interpolated_in_XY[qy][qx].gradient[1] * interpolate_in_Z;
            u_q[qz][qy][qx].gradient[2] += interpolated_in_XY[qy][qx].value       * differentiate_in_Z;
          }
        }
      }

    }

    return u_q;

  }

}

template < typename lambda, typename T, int ... n, Geometry geom, int q >
auto BatchApply(lambda qf, tensor< T, n ... > qf_inputs, GaussLegendreRule<geom, q> rule, mfem::DeviceTensor< 6, const double > J_q, int e) {

  if constexpr (geom == Geometry::Hexahedron) {

    constexpr int dim = 3;

    using output_type = decltype(qf(qf_inputs[0][0][0]));

    tensor< output_type, q, q, q > qf_outputs;

    int q_id = 0;
    for (int qz = 0; qz < q; ++qz) {
      for (int qy = 0; qy < q; ++qy) {
        for (int qx = 0; qx < q; ++qx) {

          auto qf_input = qf_inputs[qz][qy][qx];
          
          auto J = make_tensor<dim, dim>([&](int i, int j) { return J_q(qx, qy, qz, i, j, e); });
          auto invJ = inv(J);
          auto dv = det(J) * rule.weight(qx, qy, qz);

          qf_input.gradient = dot(qf_input.gradient, invJ);

          qf_outputs[qz][qy][qx] = qf(qf_input) * dv;

          serac::get<1>(qf_outputs[qz][qy][qx]) = dot(invJ, serac::get<1>(qf_outputs[qz][qy][qx]));

          q_id++;

        }
      }
    }

    return qf_outputs;

  }

}

template < typename trial_space, typename T, Geometry geom, int q >
auto BatchPostprocess(const tensor < T, q, q, q > qf_outputs, GaussLegendreRule<geom, q> rule) {

  if constexpr (geom == Geometry::Hexahedron) {

    static constexpr int n = trial_space::order + 1;

    tensor< double, q, n > B{};
    tensor< double, q, n > G{};
    for (int i = 0; i < q; i++) {
      B[i] = GaussLobattoInterpolation<n>(rule.points_1D[i]);
      G[i] = GaussLobattoInterpolationDerivative<n>(rule.points_1D[i]);
    }

    tensor< double, n, n, n > element_residual{};

    for (int qz = 0; qz < q; ++qz) {
      tensor < value_and_gradient< double, tensor< double, 3 > >, n, n > gradXY{};
      for (int qy = 0; qy < q; ++qy) {
        tensor < value_and_gradient< double, tensor< double, 3 > >, n > gradX{};
        for (int qx = 0; qx < q; ++qx) {
          const T qf_output = qf_outputs[qz][qy][qx];
          for (int dx = 0; dx < n; ++dx) {
            const double wx = B(qx, dx);
            const double wDx = G(qx, dx);
            gradX[dx].value       += serac::get<0>(qf_output) * wx;
            gradX[dx].gradient[0] += serac::get<1>(qf_output)[0] * wDx;
            gradX[dx].gradient[1] += serac::get<1>(qf_output)[1] * wx;
            gradX[dx].gradient[2] += serac::get<1>(qf_output)[2] * wx;
          }
        }
        for (int dy = 0; dy < n; ++dy) {
          const double wy = B(qy, dy);
          const double wDy = G(qy, dy);
          for (int dx = 0; dx < n; ++dx) {
            gradXY[dy][dx].value       += gradX[dx].value       * wy;
            gradXY[dy][dx].gradient[0] += gradX[dx].gradient[0] * wy;
            gradXY[dy][dx].gradient[1] += gradX[dx].gradient[1] * wDy;
            gradXY[dy][dx].gradient[2] += gradX[dx].gradient[2] * wy;
          }
        }
      }
      for (int dz = 0; dz < n; ++dz) {
        const double wz = B(qz, dz);
        const double wDz = G(qz, dz);
        for (int dy = 0; dy < n; ++dy) {
          for (int dx = 0; dx < n; ++dx) {
            auto tmp = gradXY[dy][dx];
            element_residual[dx][dy][dz] += (tmp.value + tmp.gradient[0] + tmp.gradient[1]) * wz + tmp.gradient[2] * wDz;
          }
        }
      }
    }

    return element_residual;

  }

}

namespace detail {

template <int n>
SERAC_HOST_DEVICE void Add(const mfem::DeviceTensor<4, double>& r_global, const tensor<double, n, n, n> r_elem, int e)
{
  for (int ix = 0; ix < n; ix++) {
    for (int iy = 0; iy < n; iy++) {
      for (int iz = 0; iz < n; iz++) {
        r_global(ix, iy, iz, e) += r_elem[ix][iy][iz];
      }
    }
  }
}

} // namespace detail

template < Geometry geom, typename test, typename trial, int Q, typename lambda >
void batched_kernel(const mfem::Vector & U_, mfem::Vector & R_, const mfem::Vector & J_, size_t num_elements_, lambda qf_) {

  using trial_element              = finite_element<geom, trial>;
  using test_element               = finite_element<geom, test>;
  static constexpr int  dim        = dimension_of(geom);
  static constexpr int  test_n     = test_element::order + 1;
  static constexpr int  trial_n    = trial_element::order + 1;
  static constexpr auto rule       = GaussLegendreRule<geom, Q>();

  // mfem provides this information in 1D arrays, so we reshape it
  // into strided multidimensional arrays before using
  auto J = mfem::Reshape(J_.HostRead(), Q, Q, Q, dim, dim, num_elements_);
  auto r = mfem::Reshape(R_.HostReadWrite(), test_n, test_n, test_n, int(num_elements_));
  auto u = mfem::Reshape(U_.HostRead(), trial_n, trial_n, trial_n, int(num_elements_));

  // for each element in the domain
  for (uint32_t e = 0; e < num_elements_; e++) {

    auto args = BatchPreprocess<trial>(u, rule, e);

    auto qf_outputs = BatchApply(qf_, args, rule, J, e);

    auto r_elem = BatchPostprocess<test>(qf_outputs, rule);

    detail::Add(r, r_elem, int(e));

  }

}

template <Geometry g, typename test, typename trial, int Q, typename lambda>
__global__ void reference_cuda_kernel(mfem::DeviceTensor< 2, const double > u, 
                                      mfem::DeviceTensor< 2, double > r, 
                                      mfem::DeviceTensor< 4, const double > J, 
                                      size_t num_elements, 
                                      lambda qf) {

  using test_element          = finite_element<g, test>;
  using trial_element         = finite_element<g, trial>;
  using element_residual_type = typename test_element::residual_type;
  static constexpr auto rule  = GaussQuadratureRule<g, Q>();
  static constexpr int  dim   = dimension_of(g);

  const int grid_stride = blockDim.x * gridDim.x;

  for (int qe = blockIdx.x * blockDim.x + threadIdx.x; qe < num_elements * rule.size(); qe += grid_stride) {

    int e = qe / rule.size();
    int q = qe % rule.size();

    auto u_elem = detail::Load<trial_element>(u, e);

    element_residual_type r_elem{};

    auto   xi  = rule.points[q];
    auto   dxi = rule.weights[q];
    auto   J_q = make_tensor<dim, dim>([&](int i, int j) { return J(q, i, j, e); });
    double dx  = det(J_q) * dxi;

    auto arg = domain_integral::Preprocess<trial_element>(u_elem, xi, J_q);

    auto qf_output = qf(arg);

    r_elem += domain_integral::Postprocess<test_element>(qf_output, xi, J_q) * dx;

    detail::Add(r, r_elem, e);

  }

}

template < typename lambda, typename T, int ... n, Geometry geom, int q >
__device__ auto BatchApplyCUDA(lambda qf, T qf_input, GaussLegendreRule<geom, q> rule, mfem::DeviceTensor< 6, const double > J_q, int e) {

  if constexpr (geom == Geometry::Hexahedron) {

    constexpr int dim = 3;

    auto J = make_tensor<dim, dim>([&](int i, int j) { return J_q(threadIdx.x, threadIdx.y, threadIdx.z, i, j, e); });

    auto invJ = inv(J);

    auto dv = det(J) * rule.weight(threadIdx.x, threadIdx.y, threadIdx.z);

    qf_input.gradient = dot(qf_input.gradient, invJ);

    auto qf_output = qf(qf_input) * dv;

    serac::get<1>(qf_output) = dot(invJ, serac::get<1>(qf_output));

    return qf_output;

  }

}

template <Geometry g, typename test, typename trial, int Q, typename lambda>
__global__ void batched_cuda_kernel(mfem::DeviceTensor< 4, const double > u, 
                                    mfem::DeviceTensor< 4, double > r, 
                                    mfem::DeviceTensor< 6, const double > J, 
                                    size_t num_elements, 
                                    lambda qf) {

  static constexpr auto rule  = GaussLegendreRule<g, Q>();

  // for each element in the domain
  uint32_t e = blockIdx.x;

  // interpolate each quadrature point's value
  auto qf_input = BatchPreprocessCUDA<trial>(u, rule, e);

  // evalute the q-function
  auto qf_output = BatchApplyCUDA(qf, qf_input, rule, J, e);

  // integrate the material response against the test-space basis functions
  BatchPostprocessCUDA<test>(qf_output, rule, r, e);

}

template < typename lambda, typename T, int ... n, Geometry geom, int q >
__device__ auto BatchApplyCUDA_with_cache(lambda qf, T qf_input, GaussLegendreRule<geom, q> rule, mfem::DeviceTensor< 6, const double > J_q, int e, tensor< double, 4, q, q, q > & cache) {

  if constexpr (geom == Geometry::Hexahedron) {

    constexpr int dim = 3;

    auto J = make_tensor<dim, dim>([&](int i, int j) { return J_q(threadIdx.x, threadIdx.y, threadIdx.z, i, j, e); });

    auto invJ = inv(J);

    auto dv = det(J) * rule.weight(threadIdx.x, threadIdx.y, threadIdx.z);

    qf_input.gradient = dot(qf_input.gradient, invJ);

    auto qf_output = qf(qf_input) * dv;

    serac::get<1>(qf_output) = dot(invJ, serac::get<1>(qf_output));

    cache(0, threadIdx.z, threadIdx.y, threadIdx.x) = serac::get<0>(qf_output);
    cache(1, threadIdx.z, threadIdx.y, threadIdx.x) = serac::get<1>(qf_output)[0];
    cache(2, threadIdx.z, threadIdx.y, threadIdx.x) = serac::get<1>(qf_output)[1];
    cache(3, threadIdx.z, threadIdx.y, threadIdx.x) = serac::get<1>(qf_output)[2];

  }

}

template <Geometry g, typename test, typename trial, int q, typename lambda>
__global__ void batched_cuda_kernel_with_cache(mfem::DeviceTensor< 4, const double > u, 
                                               mfem::DeviceTensor< 4, double > r, 
                                               mfem::DeviceTensor< 6, const double > J, 
                                               size_t num_elements, 
                                               lambda qf) {

  static constexpr int n = trial::order + 1;
  static constexpr auto rule  = GaussLegendreRule<g, q>();

  __shared__ tensor< double, q, n > B;
  __shared__ tensor< double, q, n > G;

  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    for (int i = 0; i < q; i++) {
      B[i] = GaussLobattoInterpolation<n>(rule.point(i));
      G[i] = GaussLobattoInterpolationDerivative<n>(rule.point(i));
    }
  }

  __shared__ tensor < double, n, n, n > u_elem;
  __shared__ tensor < double, 3, n, n, q > A1;
  __shared__ tensor < double, 3, n, q, q > A2;

  __shared__ tensor < double, 4, q, q, q > qf_output;
  __shared__ tensor < double, 3, q, q, n > A3;
  __shared__ tensor < double, 2, q, n, n > A4;

  // for each element in the domain
  uint32_t e = blockIdx.x;

  for (int dz = threadIdx.z; dz < n; dz += blockDim.z) {
    for (int dy = threadIdx.y; dy < n; dy += blockDim.y) {
      for (int dx = threadIdx.x; dx < n; dx += blockDim.x) {
        u_elem(dz, dy, dx) = u(dx, dy, dz, e);
      }
    }
  }
  __syncthreads(); 

  // interpolate each quadrature point's value
  auto qf_input = BatchPreprocessCUDA<trial>(u_elem, rule, B, G, A1, A2);

  // evalute the q-function
  BatchApplyCUDA_with_cache(qf, qf_input, rule, J, e, qf_output);

  // integrate the material response against the test-space basis functions
  BatchPostprocessCUDA<test>(qf_output, rule, r, e, B, G, A3, A4);

}

template <Geometry g, typename test, typename trial, int q, typename lambda>
__global__ void batched_cuda_kernel_with_union_cache(mfem::DeviceTensor< 4, const double > u, 
                                                     mfem::DeviceTensor< 4, double > r, 
                                                     mfem::DeviceTensor< 6, const double > J, 
                                                     size_t num_elements, 
                                                     lambda qf) {

  static constexpr int n = trial::order + 1;
  static constexpr auto rule  = GaussLegendreRule<g, q>();

  __shared__ tensor< double, q, n > B;
  __shared__ tensor< double, q, n > G;

  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    for (int i = 0; i < q; i++) {
      B[i] = GaussLobattoInterpolation<n>(rule.point(i));
      G[i] = GaussLobattoInterpolationDerivative<n>(rule.point(i));
    }
  }

  __shared__ union {
    tensor < double, n, n, n > u_elem;
    tensor < double, 3, n, q, q > A2;
    tensor < double, 2, q, n, n > A4;
  } cache1;

  __shared__ union {
    tensor < double, 3, n, n, q > A1;
    tensor < double, 4, q, q, q > qf_output;
    tensor < double, 3, q, q, n > A3;
  } cache2;

  // for each element in the domain
  uint32_t e = blockIdx.x;

  // load the values for that element
  for (int dz = threadIdx.z; dz < n; dz += blockDim.z) {
    for (int dy = threadIdx.y; dy < n; dy += blockDim.y) {
      for (int dx = threadIdx.x; dx < n; dx += blockDim.x) {
        cache1.u_elem(dz, dy, dx) = u(dx, dy, dz, e);
      }
    }
  }
  __syncthreads(); 

  // interpolate each quadrature point's value
  auto qf_input = BatchPreprocessCUDA<trial>(cache1.u_elem, rule, B, G, cache2.A1, cache1.A2);

  // evalute the q-function
  BatchApplyCUDA_with_cache(qf, qf_input, rule, J, e, cache2.qf_output);

  // integrate the material response against the test-space basis functions
  BatchPostprocessCUDA<test>(cache2.qf_output, rule, r, e, B, G, cache2.A3, cache1.A4);

}

template <Geometry g, typename test, typename trial, int q, int n, typename lambda>
__global__ void batched_cuda_kernel_with_union_cache_const_memory(mfem::DeviceTensor< 4, const double > u, 
                                                     mfem::DeviceTensor< 4, double > r, 
                                                     mfem::DeviceTensor< 6, const double > J, 
                                                     const tensor< double, q, n > B,
                                                     const tensor< double, q, n > G,
                                                     size_t num_elements, 
                                                     lambda qf) {

  static constexpr auto rule  = GaussLegendreRule<g, q>();

  __shared__ union {
    tensor < double, n, n, n > u_elem;
    tensor < double, 3, n, q, q > A2;
    tensor < double, 2, q, n, n > A4;
  } cache1;

  __shared__ union {
    tensor < double, 3, n, n, q > A1;
    tensor < double, 4, q, q, q > qf_output;
    tensor < double, 3, q, q, n > A3;
  } cache2;

  // for each element in the domain
  uint32_t e = blockIdx.x;

  for (int dz = threadIdx.z; dz < n; dz += blockDim.z) {
    for (int dy = threadIdx.y; dy < n; dy += blockDim.y) {
      for (int dx = threadIdx.x; dx < n; dx += blockDim.x) {
        cache1.u_elem(dz, dy, dx) = u(dx, dy, dz, e);
      }
    }
  }
  __syncthreads(); 

  // interpolate each quadrature point's value
  auto qf_input = BatchPreprocessCUDA<trial>(cache1.u_elem, rule, B, G, cache2.A1, cache1.A2);

  // evalute the q-function
  BatchApplyCUDA_with_cache(qf, qf_input, rule, J, e, cache2.qf_output);

  // integrate the material response against the test-space basis functions
  BatchPostprocessCUDA<test>(cache2.qf_output, rule, r, e, B, G, cache2.A3, cache1.A4);

}

template < typename lambda, typename T, int q >
__device__ auto batch_apply_qf(lambda qf, T qf_input, TensorProductQuadratureRule<q> rule, mfem::DeviceTensor< 6, const double > J_q, int e, tensor< double, 1, q, q, q > & cache_source, tensor< double, 3, 1, q, q, q > & cache_flux) {

  constexpr int dim = 3;

  auto J = make_tensor<dim, dim>([&](int i, int j) { return J_q(threadIdx.x, threadIdx.y, threadIdx.z, i, j, e); });

  auto invJ = inv(J);

  auto dv = det(J) * rule.weight(threadIdx.x, threadIdx.y, threadIdx.z);

  serac::get<1>(qf_input) = dot(serac::get<1>(qf_input), invJ);

  auto [source, flux] = qf(qf_input) * dv;

  flux = dot(flux, transpose(invJ));

  cache_source(0, threadIdx.z, threadIdx.y, threadIdx.x) = source;

  cache_flux(0, 0, threadIdx.z, threadIdx.y, threadIdx.x) = flux[0][0];
  cache_flux(1, 0, threadIdx.z, threadIdx.y, threadIdx.x) = flux[0][1];
  cache_flux(2, 0, threadIdx.z, threadIdx.y, threadIdx.x) = flux[0][2];
  __syncthreads();

}

template <Geometry g, typename test, typename trial, int q, typename lambda>
__global__ void batched_cuda_kernel(mfem::DeviceTensor< 5, const double > u, 
                                    mfem::DeviceTensor< 5, double > r, 
                                    mfem::DeviceTensor< 6, const double > J, 
                                    TensorProductQuadratureRule<q> rule,
                                    size_t num_elements, 
                                    lambda qf) {

  static constexpr int n = trial::order + 1;
  using test_element = finite_element<g, test>;
  using trial_element = finite_element<g, trial>;

  __shared__ union {
    tensor < double, trial::components, n, n, n > u_elem;
    tensor < double, 3, n, q, q > A2;
    tensor < double, 2, q, n, n > A4;
  } cache1;

  __shared__ union {
    tensor < double, 2, n, n, q > A1;
    struct {
      tensor < double, 1, q, q, q > source;
      tensor < double, 3, 1, q, q, q > flux;
    };
    tensor < double, 3, q, q, n > A3;
  } cache2;


  // for each element in the domain
  uint32_t e = blockIdx.x;

  for (int i = 0; i < trial::components; i++) {
    for (int dz = threadIdx.z; dz < n; dz += blockDim.z) {
      for (int dy = threadIdx.y; dy < n; dy += blockDim.y) {
        for (int dx = threadIdx.x; dx < n; dx += blockDim.x) {
          cache1.u_elem(i, dz, dy, dx) = u(dx, dy, dz, i, e);
        }
      }
    }
  }
  __syncthreads(); 

  // interpolate each quadrature point's value
  auto qf_input = trial_element::interpolate(cache1.u_elem, rule, cache2.A1, cache1.A2);

  // evalute the q-function at each quadrature point
  batch_apply_qf(qf, qf_input, rule, J, e, cache2.source, cache2.flux);

  // integrate the material response against the test-space basis functions
  test_element::extrapolate(cache2.source, cache2.flux, rule, r, e, cache2.A3, cache1.A4);

}

} // namespace serac

namespace compiler {
  static void please_do_not_optimize_away([[maybe_unused]] void* p) { asm volatile("" : : "g"(p) : "memory"); }
}

struct MassAndDiffusionQFunction {
  template < typename T >
  SERAC_HOST_DEVICE auto operator()(T input) {
    auto [u, du_dx] = input;
    auto source = rho * u;
    auto flux = k * du_dx;
    return serac::tuple{source, flux};
  }

  double rho;
  double k;
};

template <typename lambda>
auto time(lambda&& f)
{
  axom::utilities::Timer stopwatch;
  stopwatch.start();
  f();
  stopwatch.stop();
  return stopwatch.elapsed();
}

int main() {

  using serac::H1;
  using serac::Geometry;

  mfem::Device device("cuda");

  constexpr int p = 3;
  constexpr int n = p + 1;
  constexpr int q = n;
  constexpr int dim = 3;
  int num_runs = 10;
  int num_elements = 100000;

  double rho = 1.0;
  double k = 1.0;

  MassAndDiffusionQFunction qfunc{rho, k};

  using test = H1<p>;
  using trial = H1<p>;

  std::default_random_engine generator{0};
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);

  mfem::Vector U1D(num_elements * n * n * n);
  mfem::Vector R1D(num_elements * n * n * n);
  mfem::Vector J1D(num_elements * dim * dim * q * q * q);
  mfem::Vector rho_dv_1D(num_elements * q * q * q);
  mfem::Vector k_invJ_invJT_dv_1D(num_elements * dim * dim * q * q * q);

  U1D.UseDevice(true);
  R1D.UseDevice(true);
  J1D.UseDevice(true);
  rho_dv_1D.UseDevice(true);
  k_invJ_invJT_dv_1D.UseDevice(true);

  auto U = mfem::Reshape(U1D.HostReadWrite(), n, n, n, num_elements);
  auto J = mfem::Reshape(J1D.HostReadWrite(), q * q * q, dim, dim, num_elements);
  auto rho_dv = mfem::Reshape(rho_dv_1D.HostReadWrite(), q * q * q, num_elements);
  auto k_invJ_invJT_dv = mfem::Reshape(k_invJ_invJT_dv_1D.HostReadWrite(), q * q * q, dim, dim, num_elements);

  serac::GaussLegendreRule<Geometry::Hexahedron, q> rule;

  for (int e = 0; e < num_elements; e++) {

    for (int ix = 0; ix < n; ix++) {
      for (int iy = 0; iy < n; iy++) {
        for (int iz = 0; iz < n; iz++) {
          U(iz, iy, ix, e) = 0.1 * distribution(generator);
        }
      }
    }

    for (int i = 0; i < q * q * q; i++) {

      serac::tensor< double, dim, dim > J_q{};

      for (int r = 0; r < dim; r++) {
        for (int c = 0; c < dim; c++) {
          J(i, r, c, e) = J_q[r][c] = (r == c) + 0.1 * distribution(generator);
        }
      }

      int qx = i % q;
      int qy = (i % (q * q)) / q;
      int qz = i / (q * q);

      double qweight = rule.weight(qx, qy, qz);
      auto invJ_invJT = dot(inv(J_q), transpose(inv(J_q)));
      double dv = det(J_q) * qweight;

      rho_dv(i, e) = rho * dv; 
      for (int r = 0; r < dim; r++) {
        for (int c = 0; c < dim; c++) {
          k_invJ_invJT_dv(i, r, c, e) = k * invJ_invJT[r][c] * dv;
        }
      }

    }

  }

  {
    R1D = 0.0;
    double runtime = time([&]() {
      for (int i = 0; i < num_runs; i++) {
        serac::batched_kernel<Geometry::Hexahedron, test, trial, q>(U1D, R1D, J1D, num_elements, qfunc);
        compiler::please_do_not_optimize_away(&R1D);
      }
    }) / n;
    std::cout << "average batched kernel time: " << runtime / num_runs << std::endl;
  }
  auto answer_reference = R1D;

  {
    R1D = 0.0;

    mfem::DeviceTensor<2, const double > u_d = mfem::Reshape(U1D.Read(), n * n * n, num_elements);
    mfem::DeviceTensor<2, double > r_d = mfem::Reshape(R1D.ReadWrite(), n * n * n, num_elements);
    mfem::DeviceTensor<4, const double > J_d = mfem::Reshape(J1D.Read(), q * q * q, dim, dim, num_elements);
    int blocksize = 128;
    int gridsize = (num_elements * q * q * q + blocksize - 1) / blocksize;
    double runtime = time([&]() {
      for (int i = 0; i < num_runs; i++) {
        serac::reference_cuda_kernel<Geometry::Hexahedron, test, trial, q><<<gridsize, blocksize>>>(u_d, r_d, J_d, num_elements, qfunc);
        compiler::please_do_not_optimize_away(&R1D);
      }
      cudaDeviceSynchronize();
    }) / n;
    std::cout << "average reference (cuda) kernel time: " << runtime / num_runs << std::endl;
  }
  auto answer_reference_cuda = R1D;
  auto error = answer_reference;
  error -= answer_reference_cuda;
  auto relative_error = error.Norml2() / answer_reference.Norml2();
  std::cout << "error: " << relative_error << std::endl;

  {
    R1D = 0.0;

    mfem::DeviceTensor<4, const double > u_d = mfem::Reshape(U1D.Read(), n, n, n, num_elements);
    mfem::DeviceTensor<4, double > r_d = mfem::Reshape(R1D.ReadWrite(), n, n, n, num_elements);
    mfem::DeviceTensor<6, const double > J_d = mfem::Reshape(J1D.Read(), q, q, q, dim, dim, num_elements);
    dim3 blocksize{q, q, q};
    int gridsize = num_elements;
    double runtime = time([&]() {
      for (int i = 0; i < num_runs; i++) {
        serac::batched_cuda_kernel<Geometry::Hexahedron, test, trial, q><<<gridsize, blocksize>>>(u_d, r_d, J_d, num_elements, qfunc);
        compiler::please_do_not_optimize_away(&R1D);
      }
      cudaDeviceSynchronize();
    }) / n;
    std::cout << "average batched (cuda) kernel time: " << runtime / num_runs << std::endl;
  }
  answer_reference_cuda = R1D;
  error = answer_reference;
  error -= answer_reference_cuda;
  relative_error = error.Norml2() / answer_reference.Norml2();
  std::cout << "error: " << relative_error << std::endl;

  {
    R1D = 0.0;

    mfem::DeviceTensor<4, const double > u_d = mfem::Reshape(U1D.Read(), n, n, n, num_elements);
    mfem::DeviceTensor<4, double > r_d = mfem::Reshape(R1D.ReadWrite(), n, n, n, num_elements);
    mfem::DeviceTensor<6, const double > J_d = mfem::Reshape(J1D.Read(), q, q, q, dim, dim, num_elements);
    dim3 blocksize{q, q, q};
    int gridsize = num_elements;
    double runtime = time([&]() {
      for (int i = 0; i < num_runs; i++) {
        serac::batched_cuda_kernel_with_cache<Geometry::Hexahedron, test, trial, q><<<gridsize, blocksize>>>(u_d, r_d, J_d, num_elements, qfunc);
        compiler::please_do_not_optimize_away(&R1D);
      }
      cudaDeviceSynchronize();
    }) / n;
    std::cout << "average batched (cuda, w/ cache) kernel time: " << runtime / num_runs << std::endl;
  }
  answer_reference_cuda = R1D;
  error = answer_reference;
  error -= answer_reference_cuda;
  relative_error = error.Norml2() / answer_reference.Norml2();
  std::cout << "error: " << relative_error << std::endl;

  {
    R1D = 0.0;

    mfem::DeviceTensor<4, const double > u_d = mfem::Reshape(U1D.Read(), n, n, n, num_elements);
    mfem::DeviceTensor<4, double > r_d = mfem::Reshape(R1D.ReadWrite(), n, n, n, num_elements);
    mfem::DeviceTensor<6, const double > J_d = mfem::Reshape(J1D.Read(), q, q, q, dim, dim, num_elements);
    dim3 blocksize{q, q, q};
    int gridsize = num_elements;
    double runtime = time([&]() {
      for (int i = 0; i < num_runs; i++) {
        serac::batched_cuda_kernel_with_union_cache<Geometry::Hexahedron, test, trial, q><<<gridsize, blocksize>>>(u_d, r_d, J_d, num_elements, qfunc);
        compiler::please_do_not_optimize_away(&R1D);
      }
      cudaDeviceSynchronize();
    }) / n;
    std::cout << "average batched (cuda, w/ union cache) kernel time: " << runtime / num_runs << std::endl;
  }
  answer_reference_cuda = R1D;
  error = answer_reference;
  error -= answer_reference_cuda;
  relative_error = error.Norml2() / answer_reference.Norml2();
  std::cout << "error: " << relative_error << std::endl;


  {
    R1D = 0.0;

    serac::tensor< double, q, n > B;
    serac::tensor< double, q, n > G;

    for (int i = 0; i < q; i++) {
      B[i] = serac::GaussLobattoInterpolation<n>(rule.point(i));
      G[i] = serac::GaussLobattoInterpolationDerivative<n>(rule.point(i));
    }

    mfem::DeviceTensor<4, const double > u_d = mfem::Reshape(U1D.Read(), n, n, n, num_elements);
    mfem::DeviceTensor<4, double > r_d = mfem::Reshape(R1D.ReadWrite(), n, n, n, num_elements);
    mfem::DeviceTensor<6, const double > J_d = mfem::Reshape(J1D.Read(), q, q, q, dim, dim, num_elements);
    dim3 blocksize{q, q, q};
    int gridsize = num_elements;
    double runtime = time([&]() {
      for (int i = 0; i < num_runs; i++) {
        serac::batched_cuda_kernel_with_union_cache_const_memory<Geometry::Hexahedron, test, trial, q><<<gridsize, blocksize>>>(u_d, r_d, J_d, B, G, num_elements, qfunc);
        compiler::please_do_not_optimize_away(&R1D);
      }
      cudaDeviceSynchronize();
    }) / n;
    std::cout << "average batched (cuda, w/ union cacheu, B/G in __constant__ memory) kernel time: " << runtime / num_runs << std::endl;
  }
  answer_reference_cuda = R1D;
  error = answer_reference;
  error -= answer_reference_cuda;
  relative_error = error.Norml2() / answer_reference.Norml2();
  std::cout << "error: " << relative_error << std::endl;

  {
    R1D = 0.0;

    mfem::DeviceTensor<5, const double > u_d = mfem::Reshape(U1D.Read(), n, n, n, 1, num_elements);
    mfem::DeviceTensor<5, double > r_d = mfem::Reshape(R1D.ReadWrite(), n, n, n, 1, num_elements);
    mfem::DeviceTensor<6, const double > J_d = mfem::Reshape(J1D.Read(), q, q, q, dim, dim, num_elements);
    auto rule = serac::MakeGaussLegendreRule<Geometry::Hexahedron, q>();
    dim3 blocksize{q, q, q};
    int gridsize = num_elements;
    double runtime = time([&]() {
      for (int i = 0; i < num_runs; i++) {
        serac::batched_cuda_kernel<Geometry::Hexahedron, test, trial, q><<<gridsize, blocksize>>>(u_d, r_d, J_d, rule, num_elements, qfunc);
        compiler::please_do_not_optimize_away(&R1D);
      }
      cudaDeviceSynchronize();
    }) / n;
    std::cout << "average batched (cuda, using element library) kernel time: " << runtime / num_runs << std::endl;
  }
  answer_reference_cuda = R1D;
  error = answer_reference;
  error -= answer_reference_cuda;
  relative_error = error.Norml2() / answer_reference.Norml2();
  std::cout << "error: " << relative_error << std::endl;

  {
    R1D = 0.0;
    bool symmetric = false;
    mfem::Array<double> b_(n * q);
    mfem::Array<double> bt_(n * q);
    mfem::Array<double> g_(n * q);
    mfem::Array<double> gt_(n * q);
    auto B = mfem::Reshape(b_.HostReadWrite(), q, n);
    auto Bt = mfem::Reshape(bt_.HostReadWrite(), n, q);

    auto G = mfem::Reshape(g_.HostReadWrite(), q, n);
    auto Gt = mfem::Reshape(gt_.HostReadWrite(), n, q);

    for (int i = 0; i < q; i++) {
      auto value = serac::GaussLobattoInterpolation<n>(rule.points_1D[i]);
      auto derivative = serac::GaussLobattoInterpolationDerivative<n>(rule.points_1D[i]);

      for (int j = 0; j < n; j++) {
        Bt(j, i) = B(i, j) = value[j];
        Gt(j, i) = G(i, j) = derivative[j];
      }
    }

    double mass_runtime = time([&]() {
      for (int i = 0; i < num_runs; i++) {
        mfem::PAMassApply3D<n,q>(num_elements, b_, bt_, rho_dv_1D, U1D, R1D);
        compiler::please_do_not_optimize_away(&R1D);
      }
      cudaDeviceSynchronize();
    }) / n;
    std::cout << "average mfem mass kernel time: " << mass_runtime / num_runs << std::endl;

    double diffusion_runtime = time([&]() {
      for (int i = 0; i < num_runs; i++) {
        mfem::PADiffusionApply3D<n,q>(num_elements, symmetric = false, b_, g_, bt_, gt_, k_invJ_invJT_dv_1D, U1D, R1D);
        compiler::please_do_not_optimize_away(&R1D);
      }
      cudaDeviceSynchronize();
    }) / n;
    std::cout << "average mfem diffusion kernel time: " << diffusion_runtime / num_runs << std::endl;

    std::cout << "average mfem combined kernel time: " << (mass_runtime + diffusion_runtime) / num_runs << std::endl;
    auto answer_mfem = R1D;
    auto error = answer_reference;
    error -= answer_mfem;
    auto relative_error = error.Norml2() / answer_reference.Norml2();
    std::cout << "error: " << relative_error << std::endl;

    R1D = 0.0;
    mass_runtime = time([&]() {
      for (int i = 0; i < num_runs; i++) {
        mfem::SmemPAMassApply3D<n,q>(num_elements, b_, bt_, rho_dv_1D, U1D, R1D);
        compiler::please_do_not_optimize_away(&R1D);
      }
      cudaDeviceSynchronize();
    }) / n;
    std::cout << "average mfem mass kernel (Smem) time: " << mass_runtime / num_runs << std::endl;

    diffusion_runtime = time([&]() {
      for (int i = 0; i < num_runs; i++) {
        mfem::SmemPADiffusionApply3D<n,q>(num_elements, symmetric = false, b_, g_, k_invJ_invJT_dv_1D, U1D, R1D);
        compiler::please_do_not_optimize_away(&R1D);
      }
      cudaDeviceSynchronize();
    }) / n;
    std::cout << "average mfem diffusion kernel (Smem) time: " << diffusion_runtime / num_runs << std::endl;

    std::cout << "average mfem combined kernel (Smem) time: " << (mass_runtime + diffusion_runtime) / num_runs << std::endl;
    answer_mfem = R1D;
    error = answer_reference;
    error -= answer_mfem;
    relative_error = error.Norml2() / answer_reference.Norml2();
    std::cout << "error: " << relative_error << std::endl;

  }

}
