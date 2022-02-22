#include "mfem.hpp"
#include "mfem/general/forall.hpp"

#include "axom/core/utilities/Timer.hpp"

#include "serac/infrastructure/accelerator.hpp"
#include "serac/numerics/functional/tensor.hpp"
#include "serac/numerics/functional/quadrature.hpp"
#include "serac/numerics/functional/finite_element.hpp"
#include "serac/numerics/functional/tuple_arithmetic.hpp"
#include "serac/numerics/functional/integral_utilities.hpp"

namespace mfem {

template <int T_D1D = 0, int T_Q1D = 0>
static void PAMassApply3D(const int NE, const Array<double> &b_,
                          const Array<double> &bt_, const Vector &d_,
                          const Vector &x_, Vector &y_, const int d1d = 0,
                          const int q1d = 0) {
  const int D1D = T_D1D ? T_D1D : d1d;
  const int Q1D = T_Q1D ? T_Q1D : q1d;
  static_assert(D1D <= MAX_D1D);
  static_assert(Q1D <= MAX_Q1D);
  auto B = Reshape(b_.Read(), Q1D, D1D);
  auto Bt = Reshape(bt_.Read(), D1D, Q1D);
  auto D = Reshape(d_.Read(), Q1D, Q1D, Q1D, NE);
  auto X = Reshape(x_.Read(), D1D, D1D, D1D, NE);
  auto Y = Reshape(y_.ReadWrite(), D1D, D1D, D1D, NE);
  for (int e = 0; e < NE; e++) {
    constexpr int max_D1D = T_D1D ? T_D1D : MAX_D1D;
    constexpr int max_Q1D = T_Q1D ? T_Q1D : MAX_Q1D;
    double sol_xyz[max_Q1D][max_Q1D][max_Q1D];
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
  }
}

template <int T_D1D = 0, int T_Q1D = 0>
static void PADiffusionApply3D(const int NE, const bool symmetric,
                               const Array<double> &b, const Array<double> &g,
                               const Array<double> &bt, const Array<double> &gt,
                               const Vector &d_, const Vector &x_, Vector &y_,
                               int d1d = 0, int q1d = 0) {
  const int D1D = T_D1D ? T_D1D : d1d;
  const int Q1D = T_Q1D ? T_Q1D : q1d;
  static_assert(D1D <= MAX_D1D);
  static_assert(Q1D <= MAX_Q1D);
  auto B = Reshape(b.Read(), Q1D, D1D);
  auto G = Reshape(g.Read(), Q1D, D1D);
  auto Bt = Reshape(bt.Read(), D1D, Q1D);
  auto Gt = Reshape(gt.Read(), D1D, Q1D);
  auto D = Reshape(d_.Read(), Q1D * Q1D * Q1D, symmetric ? 6 : 9, NE);
  auto X = Reshape(x_.Read(), D1D, D1D, D1D, NE);
  auto Y = Reshape(y_.ReadWrite(), D1D, D1D, D1D, NE);
  for (int e = 0; e < NE; e++) {
    constexpr int max_D1D = T_D1D ? T_D1D : MAX_D1D;
    constexpr int max_Q1D = T_Q1D ? T_Q1D : MAX_Q1D;
    double grad[max_Q1D][max_Q1D][max_Q1D][3];
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
          const double wy = B(qy, dy);
          const double wDy = G(qy, dy);
          for (int qx = 0; qx < Q1D; ++qx) {
            const double wx = gradX[qx][0];
            const double wDx = gradX[qx][1];
            gradXY[qy][qx][0] += wDx * wy;
            gradXY[qy][qx][1] += wx * wDy;
            gradXY[qy][qx][2] += wx * wy;
          }
        }
      }
      for (int qz = 0; qz < Q1D; ++qz) {
        const double wz = B(qz, dz);
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
          const int q = qx + (qy + qz * Q1D) * Q1D;

          const double O11 = D(q, 0, e);
          const double O12 = D(q, 1, e);
          const double O13 = D(q, 2, e);
          const double O21 = symmetric ? O12 : D(q, 3, e);
          const double O22 = symmetric ? D(q, 3, e) : D(q, 4, e);
          const double O23 = symmetric ? D(q, 4, e) : D(q, 5, e);
          const double O31 = symmetric ? O13 : D(q, 6, e);
          const double O32 = symmetric ? O23 : D(q, 7, e);
          const double O33 = symmetric ? D(q, 5, e) : D(q, 8, e);
          const double gradX = grad[qz][qy][qx][0];
          const double gradY = grad[qz][qy][qx][1];
          const double gradZ = grad[qz][qy][qx][2];
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
            const double wx = Bt(dx, qx);
            const double wDx = Gt(dx, qx);
            gradX[dx][0] += gX * wDx;
            gradX[dx][1] += gY * wx;
            gradX[dx][2] += gZ * wx;
          }
        }
        for (int dy = 0; dy < D1D; ++dy) {
          const double wy = Bt(dy, qy);
          const double wDy = Gt(dy, qy);
          for (int dx = 0; dx < D1D; ++dx) {
            gradXY[dy][dx][0] += gradX[dx][0] * wy;
            gradXY[dy][dx][1] += gradX[dx][1] * wDy;
            gradXY[dy][dx][2] += gradX[dx][2] * wy;
          }
        }
      }
      for (int dz = 0; dz < D1D; ++dz) {
        const double wz = Bt(dz, qz);
        const double wDz = Gt(dz, qz);
        for (int dy = 0; dy < D1D; ++dy) {
          for (int dx = 0; dx < D1D; ++dx) {
            Y(dx, dy, dz, e) +=
                ((gradXY[dy][dx][0] * wz) + (gradXY[dy][dx][1] * wz) +
                 (gradXY[dy][dx][2] * wDz));
          }
        }
      }
    }
  }
}

// Half of B and G are stored in shared to get B, Bt, G and Gt.
// Indices computation for SmemPADiffusionApply3D.
static MFEM_HOST_DEVICE inline int qi(const int q, const int d, const int Q)
{
   return (q<=d) ? q : Q-1-q;
}

static MFEM_HOST_DEVICE inline int dj(const int q, const int d, const int D)
{
   return (q<=d) ? d : D-1-d;
}

static MFEM_HOST_DEVICE inline int qk(const int q, const int d, const int Q)
{
   return (q<=d) ? Q-1-q : q;
}

static MFEM_HOST_DEVICE inline int dl(const int q, const int d, const int D)
{
   return (q<=d) ? D-1-d : d;
}

static MFEM_HOST_DEVICE inline double sign(const int q, const int d)
{
   return (q<=d) ? -1.0 : 1.0;
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
      MFEM_SHARED double sBG[MQ1*MD1];
      double (*B)[MD1] = (double (*)[MD1]) sBG;
      double (*G)[MD1] = (double (*)[MD1]) sBG;
      double (*Bt)[MQ1] = (double (*)[MQ1]) sBG;
      double (*Gt)[MQ1] = (double (*)[MQ1]) sBG;
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
               const int i = qi(qx,dy,Q1D);
               const int j = dj(qx,dy,D1D);
               const int k = qk(qx,dy,Q1D);
               const int l = dl(qx,dy,D1D);
               B[i][j] = b(qx,dy);
               G[k][l] = g(qx,dy) * sign(qx,dy);
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
                  const int i = qi(qx,dx,Q1D);
                  const int j = dj(qx,dx,D1D);
                  const int k = qk(qx,dx,Q1D);
                  const int l = dl(qx,dx,D1D);
                  const double s = sign(qx,dx);
                  const double coords = X[dz][dy][dx];
                  u += coords * B[i][j];
                  v += coords * G[k][l] * s;
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
                  const int i = qi(qy,dy,Q1D);
                  const int j = dj(qy,dy,D1D);
                  const int k = qk(qy,dy,Q1D);
                  const int l = dl(qy,dy,D1D);
                  const double s = sign(qy,dy);
                  u += DDQ1[dz][dy][qx] * B[i][j];
                  v += DDQ0[dz][dy][qx] * G[k][l] * s;
                  w += DDQ0[dz][dy][qx] * B[i][j];
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
                  const int i = qi(qz,dz,Q1D);
                  const int j = dj(qz,dz,D1D);
                  const int k = qk(qz,dz,Q1D);
                  const int l = dl(qz,dz,D1D);
                  const double s = sign(qz,dz);
                  u += DQQ0[dz][qy][qx] * B[i][j];
                  v += DQQ1[dz][qy][qx] * B[i][j];
                  w += DQQ2[dz][qy][qx] * G[k][l] * s;
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

               // const int q = qx + (qy + qz * Q1D) * Q1D;
               // const double O11 = D(q, 0, e);
               // const double O12 = D(q, 1, e);
               // const double O13 = D(q, 2, e);
               // const double O21 = symmetric ? O12 : D(q, 3, e);
               // const double O22 = symmetric ? D(q, 3, e) : D(q, 4, e);
               // const double O23 = symmetric ? D(q, 4, e) : D(q, 5, e);
               // const double O31 = symmetric ? O13 : D(q, 6, e);
               // const double O32 = symmetric ? O23 : D(q, 7, e);
               // const double O33 = symmetric ? D(q, 5, e) : D(q, 8, e);
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
         MFEM_FOREACH_THREAD(d,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               const int i = qi(q,d,Q1D);
               const int j = dj(q,d,D1D);
               const int k = qk(q,d,Q1D);
               const int l = dl(q,d,D1D);
               Bt[j][i] = b(q,d);
               Gt[l][k] = g(q,d) * sign(q,d);
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
                  const int i = qi(qx,dx,Q1D);
                  const int j = dj(qx,dx,D1D);
                  const int k = qk(qx,dx,Q1D);
                  const int l = dl(qx,dx,D1D);
                  const double s = sign(qx,dx);
                  u += QQQ0[qz][qy][qx] * Gt[l][k] * s;
                  v += QQQ1[qz][qy][qx] * Bt[j][i];
                  w += QQQ2[qz][qy][qx] * Bt[j][i];
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
                  const int i = qi(qy,dy,Q1D);
                  const int j = dj(qy,dy,D1D);
                  const int k = qk(qy,dy,Q1D);
                  const int l = dl(qy,dy,D1D);
                  const double s = sign(qy,dy);
                  u += QQD0[qz][qy][dx] * Bt[j][i];
                  v += QQD1[qz][qy][dx] * Gt[l][k] * s;
                  w += QQD2[qz][qy][dx] * Bt[j][i];
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
                  const int i = qi(qz,dz,Q1D);
                  const int j = dj(qz,dz,D1D);
                  const int k = qk(qz,dz,Q1D);
                  const int l = dl(qz,dz,D1D);
                  const double s = sign(qz,dz);
                  u += QDD0[qz][dy][dx] * Bt[j][i];
                  v += QDD1[qz][dy][dx] * Bt[j][i];
                  w += QDD2[qz][dy][dx] * Gt[l][k] * s;
               }
               y(dx,dy,dz,e) += (u + v + w);
            }
         }
      }
   });
}

}

namespace serac {

template < Geometry geom, typename test, typename trial, int Q, typename lambda >
void reference_kernel(const mfem::Vector & U_, mfem::Vector & R_, const mfem::Vector & J_, size_t num_elements_, lambda qf_) {

  using trial_element              = finite_element<geom, trial>;
  using test_element               = finite_element<geom, test>;
  using element_residual_type      = typename test_element::residual_type;
  static constexpr int  dim        = dimension_of(geom);
  static constexpr int  test_ndof  = test_element::ndof;
  static constexpr int  trial_ndof = trial_element::ndof;
  static constexpr auto rule       = GaussQuadratureRule<geom, Q>();

  // mfem provides this information in 1D arrays, so we reshape it
  // into strided multidimensional arrays before using
  auto J = mfem::Reshape(J_.Read(), rule.size(), dim, dim, num_elements_);
  auto r = detail::Reshape<test>(R_.ReadWrite(), test_ndof, int(num_elements_));
  auto u = detail::Reshape<test>(U_.Read(), trial_ndof, int(num_elements_));

  // for each element in the domain
  for (uint32_t e = 0; e < num_elements_; e++) {

    // get the DOF values for this particular element
    auto u_elem = detail::Load<trial_element>(u, e);

    // this is where we will accumulate the element residual tensor
    element_residual_type r_elem{};

    // for each quadrature point in the element
    for (int q = 0; q < static_cast<int>(rule.size()); q++) {
      auto   xi  = rule.points[q];
      auto   dxi = rule.weights[q];
      auto   J_q = make_tensor<dim, dim>([&](int i, int j) { return J(q, i, j, e); });
      double dx  = det(J_q) * dxi;

      // evaluate the value/derivatives needed for the q-function at this quadrature point
      auto arg = domain_integral::Preprocess<trial_element>(u_elem, xi, J_q);

      // integrate qf_output against test space shape functions / gradients
      // to get element residual contributions
      r_elem += domain_integral::Postprocess<test_element>(qf_(arg), xi, J_q) * dx;

    }

    // once we've finished the element integration loop, write our element residuals
    // out to memory, to be later assembled into global residuals by mfem
    detail::Add(r, r_elem, int(e));
  }

}

template < typename S, typename T >
struct value_and_gradient { S value; T gradient; };

template < Geometry g, int Q >
struct GaussLegendreRule;

template < int Q >
struct GaussLegendreRule< Geometry::Hexahedron, Q > {
  static constexpr auto points_1D = GaussLegendreNodes<Q>();
  static constexpr auto weights_1D = GaussLegendreWeights<Q>();

  static constexpr double weight(int qx, int qy, int qz) { 
    return weights_1D[qx] * weights_1D[qy] * weights_1D[qz];
  }

  static constexpr int size() { return Q * Q * Q; }

};

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

template < typename lambda, typename T, int ... n, Geometry geom, int q >
auto BatchApplySIMD(lambda qf, tensor< T, n ... > qf_inputs, GaussLegendreRule<geom, q> rule, mfem::DeviceTensor< 6, const double > J_q, int e) {

  if constexpr (geom == Geometry::Hexahedron) {

    constexpr int dim = 3;

    using output_type = decltype(qf(qf_inputs[0][0][0].data));

    tensor< output_type, q, q, q > qf_outputs;

    int q_id = 0;
    for (int qz = 0; qz < q; ++qz) {
      for (int qy = 0; qy < q; ++qy) {
        for (int qx = 0; qx < q; ++qx) {

          auto qf_input = qf_inputs[qz][qy][qx].data;
          
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

template < typename lambda, typename T, int ... n, Geometry geom, int q >
void BatchApplySIMDinout(lambda qf, tensor< T, n ... > & qf_inouts, GaussLegendreRule<geom, q> rule, mfem::DeviceTensor< 6, const double > J_q, int e) {

  if constexpr (geom == Geometry::Hexahedron) {

    constexpr int dim = 3;

    int q_id = 0;
    for (int qz = 0; qz < q; ++qz) {
      for (int qy = 0; qy < q; ++qy) {
        for (int qx = 0; qx < q; ++qx) {

          auto qf_inout = qf_inouts[qz][qy][qx].data;
          
          auto J = make_tensor<dim, dim>([&](int i, int j) { return J_q(qx, qy, qz, i, j, e); });
          auto invJ = inv(J);
          auto dv = det(J) * rule.weight(qx, qy, qz);

          qf_inout.gradient = dot(qf_inout.gradient, invJ);

          qf_inout = to_value_and_gradient(qf(qf_inout) * dv);

          qf_inout.gradient = dot(invJ, qf_inout.gradient);

          qf_inouts[qz][qy][qx].data = qf_inout;

          q_id++;

        }
      }
    }

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

template < typename trial_space, typename T, Geometry geom, int q >
auto BatchPostprocessConstexpr(const tensor < T, q, q, q > qf_outputs, GaussLegendreRule<geom, q> rule) {

  if constexpr (geom == Geometry::Hexahedron) {

    static constexpr int n = trial_space::order + 1;

    static constexpr auto B = [&](){
      tensor< double, q, n > B_{};
      for (int i = 0; i < q; i++) {
        B_[i] = GaussLobattoInterpolation<n>(rule.points_1D[i]);
      }
      return B_;
    }();

    static constexpr auto G = [&](){
      tensor< double, q, n > G_{};
      for (int i = 0; i < q; i++) {
        G_[i] = GaussLobattoInterpolationDerivative<n>(rule.points_1D[i]);
      }
      return G_;
    }();

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
  auto J = mfem::Reshape(J_.Read(), Q, Q, Q, dim, dim, num_elements_);
  auto r = mfem::Reshape(R_.ReadWrite(), test_n, test_n, test_n, int(num_elements_));
  auto u = mfem::Reshape(U_.Read(), trial_n, trial_n, trial_n, int(num_elements_));

  // for each element in the domain
  for (uint32_t e = 0; e < num_elements_; e++) {

    auto args = BatchPreprocess<trial>(u, rule, e);

    auto qf_outputs = BatchApply(qf_, args, rule, J, e);

    auto r_elem = BatchPostprocess<test>(qf_outputs, rule);

    detail::Add(r, r_elem, int(e));

  }

}

} // namespace serac

namespace compiler {
  static void please_do_not_optimize_away([[maybe_unused]] void* p) { asm volatile("" : : "g"(p) : "memory"); }
}

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

  constexpr int p = 3;
  constexpr int n = p + 1;
  constexpr int q = n;
  constexpr int dim = 3;
  int num_runs = 10;
  int num_elements = 10000;

  double rho = 1.0;
  double k = 1.0;

  using test = H1<p>;
  using trial = H1<p>;

  auto mass_plus_diffusion = [=](auto input){ 
    auto [u, du_dx] = input;
    auto source = rho * u;
    auto flux = k * du_dx;
    return serac::tuple{source, flux};
  };

  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);

  mfem::Vector U1D(num_elements * n * n * n);
  mfem::Vector R1D(num_elements * n * n * n);
  mfem::Vector J1D(num_elements * dim * dim * q * q * q);
  mfem::Vector rho_dv_1D(num_elements * q * q * q);
  mfem::Vector k_invJ_invJT_dv_1D(num_elements * dim * dim * q * q * q);

  auto U = mfem::Reshape(U1D.ReadWrite(), n, n, n, num_elements);
  auto J = mfem::Reshape(J1D.ReadWrite(), q * q * q, dim, dim, num_elements);
  auto rho_dv = mfem::Reshape(rho_dv_1D.ReadWrite(), q * q * q, num_elements);
  auto k_invJ_invJT_dv = mfem::Reshape(k_invJ_invJT_dv_1D.ReadWrite(), q * q * q, dim, dim, num_elements);

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
        serac::reference_kernel<Geometry::Hexahedron, test, trial, q>(U1D, R1D, J1D, num_elements, mass_plus_diffusion);
        compiler::please_do_not_optimize_away(&R1D);
      }
    }) / n;
    std::cout << "average reference kernel time: " << runtime / num_runs << std::endl;
  }
  auto answer_reference = R1D;

  {
    R1D = 0.0;
    double runtime = time([&]() {
      for (int i = 0; i < num_runs; i++) {
        serac::batched_kernel<Geometry::Hexahedron, test, trial, q>(U1D, R1D, J1D, num_elements, mass_plus_diffusion);
        compiler::please_do_not_optimize_away(&R1D);
      }
    }) / n;
    std::cout << "average batched kernel time: " << runtime / num_runs << std::endl;
  }
  auto answer_batched = R1D;
  mfem::Vector error = answer_reference;
  error -= answer_batched;
  double relative_error = error.Norml2() / answer_reference.Norml2();
  std::cout << "error: " << relative_error << std::endl;


  {
    R1D = 0.0;
    bool symmetric = false;
    mfem::Array<double> b_(n * q);
    mfem::Array<double> bt_(n * q);
    mfem::Array<double> g_(n * q);
    mfem::Array<double> gt_(n * q);
    auto B = mfem::Reshape(b_.ReadWrite(), q, n);
    auto Bt = mfem::Reshape(bt_.ReadWrite(), n, q);

    auto G = mfem::Reshape(g_.ReadWrite(), q, n);
    auto Gt = mfem::Reshape(gt_.ReadWrite(), n, q);

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
        //mfem::PAMassApply3D<n,q>(num_elements, b_, bt_, rho_dv_1D, U1D, R1D);
        mfem::SmemPAMassApply3D<n,q>(num_elements, b_, bt_, rho_dv_1D, U1D, R1D);
        compiler::please_do_not_optimize_away(&R1D);
      }
    }) / n;
    std::cout << "average mfem mass kernel time: " << mass_runtime / num_runs << std::endl;

    double diffusion_runtime = time([&]() {
      for (int i = 0; i < num_runs; i++) {
        //mfem::PADiffusionApply3D<n,q>(num_elements, symmetric = false, b_, g_, bt_, gt_, k_invJ_invJT_dv_1D, U1D, R1D);
        mfem::SmemPADiffusionApply3D<n,q>(num_elements, symmetric = false, b_, g_, k_invJ_invJT_dv_1D, U1D, R1D);
        compiler::please_do_not_optimize_away(&R1D);
      }
    }) / n;
    std::cout << "average mfem diffusion kernel time: " << diffusion_runtime / num_runs << std::endl;

    std::cout << "average mfem combined kernel time: " << (mass_runtime + diffusion_runtime) / num_runs << std::endl;
  }
  auto answer_mfem = R1D;
  error = answer_reference;
  error -= answer_mfem;
  relative_error = error.Norml2() / answer_reference.Norml2();
  std::cout << "error: " << relative_error << std::endl;

}
