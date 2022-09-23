#include "mfem.hpp"
#include "mfem/general/forall.hpp"

namespace mfem {

template<int T_D1D = 0, int T_Q1D = 0, int T_NBZ = 0>
static void SmemPAMassApply2D(const int NE,
                              const Array<double> &b_,
                              const Array<double> &bt_,
                              const Vector &d_,
                              const Vector &x_,
                              Vector &y_,
                              const int d1d = 0,
                              const int q1d = 0)
{
   MFEM_CONTRACT_VAR(bt_);
   const int D1D_ = T_D1D ? T_D1D : d1d;
   const int Q1D_ = T_Q1D ? T_Q1D : q1d;
   constexpr int NBZ_ = T_NBZ ? T_NBZ : 1;
   constexpr int MQ1_ = T_Q1D ? T_Q1D : MAX_Q1D;
   constexpr int MD1_ = T_D1D ? T_D1D : MAX_D1D;
   MFEM_VERIFY(D1D_ <= MD1_, "");
   MFEM_VERIFY(Q1D_ <= MQ1_, "");
   auto b = Reshape(b_.Read(), Q1D_, D1D_);
   auto D = Reshape(d_.Read(), Q1D_, Q1D_, NE);
   auto x = Reshape(x_.Read(), D1D_, D1D_, NE);
   auto Y = Reshape(y_.ReadWrite(), D1D_, D1D_, NE);
   MFEM_FORALL_2D(e, NE, Q1D_, Q1D_, NBZ_,
   {
      const int tidz = MFEM_THREAD_ID(z);
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int NBZ = T_NBZ ? T_NBZ : 1;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
      constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;
      MFEM_SHARED double BBt[MQ1*MD1];
      double (*B)[MD1]  = reinterpret_cast< double (*)[MD1] >(BBt);
      double (*Bt)[MQ1] = reinterpret_cast< double (*)[MQ1] >(BBt);
      MFEM_SHARED double sm0[NBZ][MDQ*MDQ];
      MFEM_SHARED double sm1[NBZ][MDQ*MDQ];
      double (*X)[MD1]  = reinterpret_cast< double (*)[MD1]>(sm0 + tidz);
      double (*DQ)[MQ1] = reinterpret_cast< double (*)[MQ1]>(sm1 + tidz);
      double (*QQ)[MQ1] = reinterpret_cast< double (*)[MQ1]>(sm0 + tidz);
      double (*QD)[MD1] = reinterpret_cast< double (*)[MD1]>(sm1 + tidz);
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            X[dy][dx] = x(dx,dy,e);
         }
      }
      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               B[q][dy] = b(q,dy);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double dq = 0.0;
            for (int dx = 0; dx < D1D; ++dx)
            {
               dq += X[dy][dx] * B[qx][dx];
            }
            DQ[dy][qx] = dq;
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double qq = 0.0;
            for (int dy = 0; dy < D1D; ++dy)
            {
               qq += DQ[dy][qx] * B[qy][dy];
            }
            QQ[qy][qx] = qq * D(qx, qy, e);
         }
      }
      MFEM_SYNC_THREAD;
      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               Bt[dy][q] = b(q,dy);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            double dq = 0.0;
            for (int qx = 0; qx < Q1D; ++qx)
            {
               dq += QQ[qy][qx] * Bt[dx][qx];
            }
            QD[qy][dx] = dq;
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            double dd = 0.0;
            for (int qy = 0; qy < Q1D; ++qy)
            {
               dd += (QD[qy][dx] * Bt[dy][qy]);
            }
            Y(dx, dy, e) += dd;
         }
      }
   });
}

// Shared memory PA Diffusion Apply 2D kernel
template<int T_D1D = 0, int T_Q1D = 0, int T_NBZ = 0>
static void SmemPADiffusionApply2D(const int NE,
                                   const bool symmetric,
                                   const Array<double> &b_,
                                   const Array<double> &g_,
                                   const Vector &d_,
                                   const Vector &x_,
                                   Vector &y_,
                                   const int d1d = 0,
                                   const int q1d = 0)
{
   const int D1D_ = T_D1D ? T_D1D : d1d;
   const int Q1D_ = T_Q1D ? T_Q1D : q1d;
   constexpr int NBZ_ = T_NBZ ? T_NBZ : 1;
   constexpr int MQ1_ = T_Q1D ? T_Q1D : MAX_Q1D;
   constexpr int MD1_ = T_D1D ? T_D1D : MAX_D1D;
   MFEM_VERIFY(D1D_ <= MD1_, "");
   MFEM_VERIFY(Q1D_ <= MQ1_, "");
   auto b = Reshape(b_.Read(), Q1D_, D1D_);
   auto g = Reshape(g_.Read(), Q1D_, D1D_);
   auto D = Reshape(d_.Read(), Q1D_*Q1D_, symmetric ? 3 : 4, NE);
   auto x = Reshape(x_.Read(), D1D_, D1D_, NE);
   auto Y = Reshape(y_.ReadWrite(), D1D_, D1D_, NE);
   MFEM_FORALL_2D(e, NE, Q1D_, Q1D_, NBZ_,
   {
      const int tidz = MFEM_THREAD_ID(z);
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int NBZ = T_NBZ ? T_NBZ : 1;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
      MFEM_SHARED double sBG[2][MQ1*MD1];
      double (*B)[MD1]  = reinterpret_cast< double (*)[MD1] >(sBG+0);
      double (*G)[MD1]  = reinterpret_cast< double (*)[MD1] >(sBG+1);
      double (*Bt)[MQ1] = reinterpret_cast< double (*)[MQ1] >(sBG+0);
      double (*Gt)[MQ1] = reinterpret_cast< double (*)[MQ1] >(sBG+1);
      MFEM_SHARED double Xz[NBZ][MD1][MD1];
      MFEM_SHARED double GD[2][NBZ][MD1][MQ1];
      MFEM_SHARED double GQ[2][NBZ][MD1][MQ1];
      double (*X)[MD1]   = reinterpret_cast< double (*)[MD1]>(Xz + tidz);
      double (*DQ0)[MD1] = reinterpret_cast< double (*)[MD1]>(GD[0] + tidz);
      double (*DQ1)[MD1] = reinterpret_cast< double (*)[MD1]>(GD[1] + tidz);
      double (*QQ0)[MD1] = reinterpret_cast< double (*)[MD1]>(GQ[0] + tidz);
      double (*QQ1)[MD1] = reinterpret_cast< double (*)[MD1]>(GQ[1] + tidz);
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            X[dy][dx] = x(dx,dy,e);
         }
      }
      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               B[q][dy] = b(q,dy);
               G[q][dy] = g(q,dy);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double u = 0.0;
            double v = 0.0;
            for (int dx = 0; dx < D1D; ++dx)
            {
               const double coords = X[dy][dx];
               u += B[qx][dx] * coords;
               v += G[qx][dx] * coords;
            }
            DQ0[dy][qx] = u;
            DQ1[dy][qx] = v;
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double u = 0.0;
            double v = 0.0;
            for (int dy = 0; dy < D1D; ++dy)
            {
               u += DQ1[dy][qx] * B[qy][dy];
               v += DQ0[dy][qx] * G[qy][dy];
            }
            QQ0[qy][qx] = u;
            QQ1[qy][qx] = v;
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            const int q = (qx + ((qy) * Q1D));
            const double O11 = D(q,0,e);
            const double O21 = D(q,1,e);
            const double O12 = symmetric ? O21 : D(q,2,e);
            const double O22 = symmetric ? D(q,2,e) : D(q,3,e);
            const double gX = QQ0[qy][qx];
            const double gY = QQ1[qy][qx];
            QQ0[qy][qx] = (O11 * gX) + (O12 * gY);
            QQ1[qy][qx] = (O21 * gX) + (O22 * gY);
         }
      }
      MFEM_SYNC_THREAD;
      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               Bt[dy][q] = b(q,dy);
               Gt[dy][q] = g(q,dy);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            double u = 0.0;
            double v = 0.0;
            for (int qx = 0; qx < Q1D; ++qx)
            {
               u += Gt[dx][qx] * QQ0[qy][qx];
               v += Bt[dx][qx] * QQ1[qy][qx];
            }
            DQ0[qy][dx] = u;
            DQ1[qy][dx] = v;
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            double u = 0.0;
            double v = 0.0;
            for (int qy = 0; qy < Q1D; ++qy)
            {
               u += DQ0[qy][dx] * Bt[dy][qy];
               v += DQ1[qy][dx] * Gt[dy][qy];
            }
            Y(dx,dy,e) += (u + v);
         }
      }
   });
}



template <int D1D, int Q1D>
static void SmemPAMassApply3D(const int NE, const Array<double>& b_, const Array<double>& bt_, const Vector& d_,
                              const Vector& x_, Vector& y_)
{
  MFEM_CONTRACT_VAR(bt_);
  auto b = Reshape(b_.Read(), Q1D, D1D);
  auto d = Reshape(d_.Read(), Q1D, Q1D, Q1D, NE);
  auto x = Reshape(x_.Read(), D1D, D1D, D1D, NE);
  auto y = Reshape(y_.ReadWrite(), D1D, D1D, D1D, NE);
  MFEM_FORALL_3D(e, NE, Q1D, Q1D, 1, {
    constexpr int      MQ1 = Q1D;
    constexpr int      MD1 = D1D;
    constexpr int      MDQ = (MQ1 > MD1) ? MQ1 : MD1;
    MFEM_SHARED double sDQ[MQ1 * MD1];
    double(*B)[MD1]  = reinterpret_cast<double(*)[MD1]>(sDQ);
    double(*Bt)[MQ1] = reinterpret_cast<double(*)[MQ1]>(sDQ);
    MFEM_SHARED double sm0[MDQ * MDQ * MDQ];
    MFEM_SHARED double sm1[MDQ * MDQ * MDQ];
    double(*X)[MD1][MD1]   = reinterpret_cast<double(*)[MD1][MD1]>(sm0);
    double(*DDQ)[MD1][MQ1] = reinterpret_cast<double(*)[MD1][MQ1]>(sm1);
    double(*DQQ)[MQ1][MQ1] = reinterpret_cast<double(*)[MQ1][MQ1]>(sm0);
    double(*QQQ)[MQ1][MQ1] = reinterpret_cast<double(*)[MQ1][MQ1]>(sm1);
    double(*QQD)[MQ1][MD1] = reinterpret_cast<double(*)[MQ1][MD1]>(sm0);
    double(*QDD)[MD1][MD1] = reinterpret_cast<double(*)[MD1][MD1]>(sm1);
    MFEM_FOREACH_THREAD(dy, y, D1D)
    {
      MFEM_FOREACH_THREAD(dx, x, D1D)
      {
        MFEM_UNROLL(MD1)
        for (int dz = 0; dz < D1D; ++dz) {
          X[dz][dy][dx] = x(dx, dy, dz, e);
        }
      }
      MFEM_FOREACH_THREAD(dx, x, Q1D) { B[dx][dy] = b(dx, dy); }
    }
    MFEM_SYNC_THREAD;
    MFEM_FOREACH_THREAD(dy, y, D1D)
    {
      MFEM_FOREACH_THREAD(qx, x, Q1D)
      {
        double u[D1D];
        MFEM_UNROLL(MD1)
        for (int dz = 0; dz < D1D; dz++) {
          u[dz] = 0;
        }
        MFEM_UNROLL(MD1)
        for (int dx = 0; dx < D1D; ++dx) {
          MFEM_UNROLL(MD1)
          for (int dz = 0; dz < D1D; ++dz) {
            u[dz] += X[dz][dy][dx] * B[qx][dx];
          }
        }
        MFEM_UNROLL(MD1)
        for (int dz = 0; dz < D1D; ++dz) {
          DDQ[dz][dy][qx] = u[dz];
        }
      }
    }
    MFEM_SYNC_THREAD;
    MFEM_FOREACH_THREAD(qy, y, Q1D)
    {
      MFEM_FOREACH_THREAD(qx, x, Q1D)
      {
        double u[D1D];
        MFEM_UNROLL(MD1)
        for (int dz = 0; dz < D1D; dz++) {
          u[dz] = 0;
        }
        MFEM_UNROLL(MD1)
        for (int dy = 0; dy < D1D; ++dy) {
          MFEM_UNROLL(MD1)
          for (int dz = 0; dz < D1D; dz++) {
            u[dz] += DDQ[dz][dy][qx] * B[qy][dy];
          }
        }
        MFEM_UNROLL(MD1)
        for (int dz = 0; dz < D1D; dz++) {
          DQQ[dz][qy][qx] = u[dz];
        }
      }
    }
    MFEM_SYNC_THREAD;
    MFEM_FOREACH_THREAD(qy, y, Q1D)
    {
      MFEM_FOREACH_THREAD(qx, x, Q1D)
      {
        double u[Q1D];
        MFEM_UNROLL(MQ1)
        for (int qz = 0; qz < Q1D; qz++) {
          u[qz] = 0;
        }
        MFEM_UNROLL(MD1)
        for (int dz = 0; dz < D1D; ++dz) {
          MFEM_UNROLL(MQ1)
          for (int qz = 0; qz < Q1D; qz++) {
            u[qz] += DQQ[dz][qy][qx] * B[qz][dz];
          }
        }
        MFEM_UNROLL(MQ1)
        for (int qz = 0; qz < Q1D; qz++) {
          QQQ[qz][qy][qx] = u[qz] * d(qx, qy, qz, e);
        }
      }
    }
    MFEM_SYNC_THREAD;
    MFEM_FOREACH_THREAD(i, y, D1D)
    {
      MFEM_FOREACH_THREAD(j, x, Q1D) { Bt[i][j] = b(j, i); }
    }
    MFEM_SYNC_THREAD;
    MFEM_FOREACH_THREAD(qy, y, Q1D)
    {
      MFEM_FOREACH_THREAD(dx, x, D1D)
      {
        double u[Q1D];
        MFEM_UNROLL(MQ1)
        for (int qz = 0; qz < Q1D; ++qz) {
          u[qz] = 0;
        }
        MFEM_UNROLL(MQ1)
        for (int qx = 0; qx < Q1D; ++qx) {
          MFEM_UNROLL(MQ1)
          for (int qz = 0; qz < Q1D; ++qz) {
            u[qz] += QQQ[qz][qy][qx] * Bt[dx][qx];
          }
        }
        MFEM_UNROLL(MQ1)
        for (int qz = 0; qz < Q1D; ++qz) {
          QQD[qz][qy][dx] = u[qz];
        }
      }
    }
    MFEM_SYNC_THREAD;
    MFEM_FOREACH_THREAD(dy, y, D1D)
    {
      MFEM_FOREACH_THREAD(dx, x, D1D)
      {
        double u[Q1D];
        MFEM_UNROLL(MQ1)
        for (int qz = 0; qz < Q1D; ++qz) {
          u[qz] = 0;
        }
        MFEM_UNROLL(MQ1)
        for (int qy = 0; qy < Q1D; ++qy) {
          MFEM_UNROLL(MQ1)
          for (int qz = 0; qz < Q1D; ++qz) {
            u[qz] += QQD[qz][qy][dx] * Bt[dy][qy];
          }
        }
        MFEM_UNROLL(MQ1)
        for (int qz = 0; qz < Q1D; ++qz) {
          QDD[qz][dy][dx] = u[qz];
        }
      }
    }
    MFEM_SYNC_THREAD;
    MFEM_FOREACH_THREAD(dy, y, D1D)
    {
      MFEM_FOREACH_THREAD(dx, x, D1D)
      {
        double u[D1D];
        MFEM_UNROLL(MD1)
        for (int dz = 0; dz < D1D; ++dz) {
          u[dz] = 0;
        }
        MFEM_UNROLL(MQ1)
        for (int qz = 0; qz < Q1D; ++qz) {
          MFEM_UNROLL(MD1)
          for (int dz = 0; dz < D1D; ++dz) {
            u[dz] += QDD[qz][dy][dx] * Bt[dz][qz];
          }
        }
        MFEM_UNROLL(MD1)
        for (int dz = 0; dz < D1D; ++dz) {
          y(dx, dy, dz, e) += u[dz];
        }
      }
    }
  });
}

template <int D1D, int Q1D>
static void SmemPADiffusionApply3D(const int NE, const bool symmetric, const Array<double>& b_, const Array<double>& g_,
                                   const Vector& d_, const Vector& x_, Vector& y_)
{
  auto b = Reshape(b_.Read(), Q1D, D1D);
  auto g = Reshape(g_.Read(), Q1D, D1D);
  auto d = Reshape(d_.Read(), Q1D, Q1D, Q1D, symmetric ? 6 : 9, NE);
  auto x = Reshape(x_.Read(), D1D, D1D, D1D, NE);
  auto y = Reshape(y_.ReadWrite(), D1D, D1D, D1D, NE);
  MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D, {
    constexpr int      MQ1 = Q1D;
    constexpr int      MD1 = D1D;
    constexpr int      MDQ = (MQ1 > MD1) ? MQ1 : MD1;
    MFEM_SHARED double sBG[2][MQ1 * MD1];
    double(*B)[MD1]  = reinterpret_cast<double(*)[MD1]>(sBG + 0);
    double(*G)[MD1]  = reinterpret_cast<double(*)[MD1]>(sBG + 1);
    double(*Bt)[MQ1] = reinterpret_cast<double(*)[MQ1]>(sBG + 0);
    double(*Gt)[MQ1] = reinterpret_cast<double(*)[MQ1]>(sBG + 1);
    MFEM_SHARED double sm0[3][MDQ * MDQ * MDQ];
    MFEM_SHARED double sm1[3][MDQ * MDQ * MDQ];
    double(*X)[MD1][MD1]    = reinterpret_cast<double(*)[MD1][MD1]>(sm0 + 2);
    double(*DDQ0)[MD1][MQ1] = reinterpret_cast<double(*)[MD1][MQ1]>(sm0 + 0);
    double(*DDQ1)[MD1][MQ1] = reinterpret_cast<double(*)[MD1][MQ1]>(sm0 + 1);
    double(*DQQ0)[MQ1][MQ1] = reinterpret_cast<double(*)[MQ1][MQ1]>(sm1 + 0);
    double(*DQQ1)[MQ1][MQ1] = reinterpret_cast<double(*)[MQ1][MQ1]>(sm1 + 1);
    double(*DQQ2)[MQ1][MQ1] = reinterpret_cast<double(*)[MQ1][MQ1]>(sm1 + 2);
    double(*QQQ0)[MQ1][MQ1] = reinterpret_cast<double(*)[MQ1][MQ1]>(sm0 + 0);
    double(*QQQ1)[MQ1][MQ1] = reinterpret_cast<double(*)[MQ1][MQ1]>(sm0 + 1);
    double(*QQQ2)[MQ1][MQ1] = reinterpret_cast<double(*)[MQ1][MQ1]>(sm0 + 2);
    double(*QQD0)[MQ1][MD1] = reinterpret_cast<double(*)[MQ1][MD1]>(sm1 + 0);
    double(*QQD1)[MQ1][MD1] = reinterpret_cast<double(*)[MQ1][MD1]>(sm1 + 1);
    double(*QQD2)[MQ1][MD1] = reinterpret_cast<double(*)[MQ1][MD1]>(sm1 + 2);
    double(*QDD0)[MD1][MD1] = reinterpret_cast<double(*)[MD1][MD1]>(sm0 + 0);
    double(*QDD1)[MD1][MD1] = reinterpret_cast<double(*)[MD1][MD1]>(sm0 + 1);
    double(*QDD2)[MD1][MD1] = reinterpret_cast<double(*)[MD1][MD1]>(sm0 + 2);
    MFEM_FOREACH_THREAD(dz, z, D1D)
    {
      MFEM_FOREACH_THREAD(dy, y, D1D)
      {
        MFEM_FOREACH_THREAD(dx, x, D1D) { X[dz][dy][dx] = x(dx, dy, dz, e); }
      }
    }
    if (MFEM_THREAD_ID(z) == 0) {
      MFEM_FOREACH_THREAD(dy, y, D1D)
      {
        MFEM_FOREACH_THREAD(qx, x, Q1D)
        {
          B[qx][dy] = b(qx, dy);
          G[qx][dy] = g(qx, dy);
        }
      }
    }
    MFEM_SYNC_THREAD;
    MFEM_FOREACH_THREAD(dz, z, D1D)
    {
      MFEM_FOREACH_THREAD(dy, y, D1D)
      {
        MFEM_FOREACH_THREAD(qx, x, Q1D)
        {
          double u = 0.0, v = 0.0;
          MFEM_UNROLL(MD1)
          for (int dx = 0; dx < D1D; ++dx) {
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
    MFEM_FOREACH_THREAD(dz, z, D1D)
    {
      MFEM_FOREACH_THREAD(qy, y, Q1D)
      {
        MFEM_FOREACH_THREAD(qx, x, Q1D)
        {
          double u = 0.0, v = 0.0, w = 0.0;
          MFEM_UNROLL(MD1)
          for (int dy = 0; dy < D1D; ++dy) {
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
    MFEM_FOREACH_THREAD(qz, z, Q1D)
    {
      MFEM_FOREACH_THREAD(qy, y, Q1D)
      {
        MFEM_FOREACH_THREAD(qx, x, Q1D)
        {
          double u = 0.0, v = 0.0, w = 0.0;
          MFEM_UNROLL(MD1)
          for (int dz = 0; dz < D1D; ++dz) {
            u += DQQ0[dz][qy][qx] * B[qz][dz];
            v += DQQ1[dz][qy][qx] * B[qz][dz];
            w += DQQ2[dz][qy][qx] * G[qz][dz];
          }
          const double O11 = d(qx, qy, qz, 0, e);
          const double O12 = d(qx, qy, qz, 1, e);
          const double O13 = d(qx, qy, qz, 2, e);
          const double O21 = symmetric ? O12 : d(qx, qy, qz, 3, e);
          const double O22 = symmetric ? d(qx, qy, qz, 3, e) : d(qx, qy, qz, 4, e);
          const double O23 = symmetric ? d(qx, qy, qz, 4, e) : d(qx, qy, qz, 5, e);
          const double O31 = symmetric ? O13 : d(qx, qy, qz, 6, e);
          const double O32 = symmetric ? O23 : d(qx, qy, qz, 7, e);
          const double O33 = symmetric ? d(qx, qy, qz, 5, e) : d(qx, qy, qz, 8, e);
          const double gX  = u;
          const double gY  = v;
          const double gZ  = w;
          QQQ0[qz][qy][qx] = (O11 * gX) + (O12 * gY) + (O13 * gZ);
          QQQ1[qz][qy][qx] = (O21 * gX) + (O22 * gY) + (O23 * gZ);
          QQQ2[qz][qy][qx] = (O31 * gX) + (O32 * gY) + (O33 * gZ);
        }
      }
    }
    MFEM_SYNC_THREAD;
    if (MFEM_THREAD_ID(z) == 0) {
      MFEM_FOREACH_THREAD(dy, y, D1D)
      {
        MFEM_FOREACH_THREAD(qx, x, Q1D)
        {
          Bt[dy][qx] = b(qx, dy);
          Gt[dy][qx] = g(qx, dy);
        }
      }
    }
    MFEM_SYNC_THREAD;
    MFEM_FOREACH_THREAD(qz, z, Q1D)
    {
      MFEM_FOREACH_THREAD(qy, y, Q1D)
      {
        MFEM_FOREACH_THREAD(dx, x, D1D)
        {
          double u = 0.0, v = 0.0, w = 0.0;
          MFEM_UNROLL(MQ1)
          for (int qx = 0; qx < Q1D; ++qx) {
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
    MFEM_FOREACH_THREAD(qz, z, Q1D)
    {
      MFEM_FOREACH_THREAD(dy, y, D1D)
      {
        MFEM_FOREACH_THREAD(dx, x, D1D)
        {
          double u = 0.0, v = 0.0, w = 0.0;
          MFEM_UNROLL(Q1D)
          for (int qy = 0; qy < Q1D; ++qy) {
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
    MFEM_FOREACH_THREAD(dz, z, D1D)
    {
      MFEM_FOREACH_THREAD(dy, y, D1D)
      {
        MFEM_FOREACH_THREAD(dx, x, D1D)
        {
          double u = 0.0, v = 0.0, w = 0.0;
          MFEM_UNROLL(MQ1)
          for (int qz = 0; qz < Q1D; ++qz) {
            u += QDD0[qz][dy][dx] * Bt[dz][qz];
            v += QDD1[qz][dy][dx] * Bt[dz][qz];
            w += QDD2[qz][dy][dx] * Gt[dz][qz];
          }
          y(dx, dy, dz, e) += (u + v + w);
        }
      }
    }
  });
}

}  // namespace mfem
