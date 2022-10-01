#include "mfem.hpp"
#include "mfem/general/forall.hpp"

namespace mfem {

template<int T_D1D = 0, int T_Q1D = 0>
static void EADiffusionAssemble2D(const int NE,
                                  const Array<double> &b,
                                  const Array<double> &g,
                                  const Vector &padata,
                                  Vector &eadata,
                                  const bool add,
                                  const int d1d = 0,
                                  const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   auto B = Reshape(b.Read(), Q1D, D1D);
   auto G = Reshape(g.Read(), Q1D, D1D);
   auto D = Reshape(padata.Read(), Q1D, Q1D, 3, NE);
   auto A = Reshape(eadata.ReadWrite(), D1D, D1D, D1D, D1D, NE);
   MFEM_FORALL_3D(e, NE, D1D, D1D, 1,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      double r_B[MQ1][MD1];
      double r_G[MQ1][MD1];
      for (int d = 0; d < D1D; d++)
      {
         for (int q = 0; q < Q1D; q++)
         {
            r_B[q][d] = B(q,d);
            r_G[q][d] = G(q,d);
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(i1,x,D1D)
      {
         MFEM_FOREACH_THREAD(i2,y,D1D)
         {
            for (int j1 = 0; j1 < D1D; ++j1)
            {
               for (int j2 = 0; j2 < D1D; ++j2)
               {
                  double val = 0.0;
                  for (int k1 = 0; k1 < Q1D; ++k1)
                  {
                     for (int k2 = 0; k2 < Q1D; ++k2)
                     {
                        double bgi = r_G[k1][i1] * r_B[k2][i2];
                        double gbi = r_B[k1][i1] * r_G[k2][i2];
                        double bgj = r_G[k1][j1] * r_B[k2][j2];
                        double gbj = r_B[k1][j1] * r_G[k2][j2];
                        double D00 = D(k1,k2,0,e);
                        double D10 = D(k1,k2,1,e);
                        double D01 = D10;
                        double D11 = D(k1,k2,2,e);
                        val += bgi * D00 * bgj + gbi * D01 * bgj + bgi * D10 * gbj + gbi * D11 * gbj;
                     }
                  }
                  if (add)
                  {
                     A(i1, i2, j1, j2, e) += val;
                  }
                  else
                  {
                     A(i1, i2, j1, j2, e) = val;
                  }
               }
            }
         }
      }
   });
}

template<int T_D1D = 0, int T_Q1D = 0>
static void EADiffusionAssemble3D(const int NE,
                                  const Array<double> &b,
                                  const Array<double> &g,
                                  const Vector &padata,
                                  Vector &eadata,
                                  const bool add,
                                  const int d1d = 0,
                                  const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   auto B = Reshape(b.Read(), Q1D, D1D);
   auto G = Reshape(g.Read(), Q1D, D1D);
   auto D = Reshape(padata.Read(), Q1D, Q1D, Q1D, 6, NE);
   auto A = Reshape(eadata.ReadWrite(), D1D, D1D, D1D, D1D, D1D, D1D, NE);
   MFEM_FORALL_3D(e, NE, D1D, D1D, D1D,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      double r_B[MQ1][MD1];
      double r_G[MQ1][MD1];
      for (int d = 0; d < D1D; d++)
      {
         for (int q = 0; q < Q1D; q++)
         {
            r_B[q][d] = B(q,d);
            r_G[q][d] = G(q,d);
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(i1,x,D1D)
      {
         MFEM_FOREACH_THREAD(i2,y,D1D)
         {
            MFEM_FOREACH_THREAD(i3,z,D1D)
            {
               for (int j1 = 0; j1 < D1D; ++j1)
               {
                  for (int j2 = 0; j2 < D1D; ++j2)
                  {
                     for (int j3 = 0; j3 < D1D; ++j3)
                     {
                        double val = 0.0;
                        for (int k1 = 0; k1 < Q1D; ++k1)
                        {
                           for (int k2 = 0; k2 < Q1D; ++k2)
                           {
                              for (int k3 = 0; k3 < Q1D; ++k3)
                              {
                                 double bbgi = r_G[k1][i1] * r_B[k2][i2] * r_B[k3][i3];
                                 double bgbi = r_B[k1][i1] * r_G[k2][i2] * r_B[k3][i3];
                                 double gbbi = r_B[k1][i1] * r_B[k2][i2] * r_G[k3][i3];
                                 double bbgj = r_G[k1][j1] * r_B[k2][j2] * r_B[k3][j3];
                                 double bgbj = r_B[k1][j1] * r_G[k2][j2] * r_B[k3][j3];
                                 double gbbj = r_B[k1][j1] * r_B[k2][j2] * r_G[k3][j3];
                                 double D00 = D(k1,k2,k3,0,e);
                                 double D10 = D(k1,k2,k3,1,e);
                                 double D20 = D(k1,k2,k3,2,e);
                                 double D01 = D10;
                                 double D11 = D(k1,k2,k3,3,e);
                                 double D21 = D(k1,k2,k3,4,e);
                                 double D02 = D20;
                                 double D12 = D21;
                                 double D22 = D(k1,k2,k3,5,e);
                                 val += bbgi * D00 * bbgj
                                        + bgbi * D10 * bbgj
                                        + gbbi * D20 * bbgj
                                        + bbgi * D01 * bgbj
                                        + bgbi * D11 * bgbj
                                        + gbbi * D21 * bgbj
                                        + bbgi * D02 * gbbj
                                        + bgbi * D12 * gbbj
                                        + gbbi * D22 * gbbj;
                              }
                           }
                        }
                        if (add)
                        {
                           A(i1, i2, i3, j1, j2, j3, e) += val;
                        }
                        else
                        {
                           A(i1, i2, i3, j1, j2, j3, e) = val;
                        }
                     }
                  }
               }
            }
         }
      }
   });
}

template<int T_D1D = 0, int T_Q1D = 0>
static void EAMassAssemble2D(const int NE,
                             const Array<double> &basis,
                             const Vector &padata,
                             Vector &eadata,
                             const bool add,
                             const int d1d = 0,
                             const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   auto B = Reshape(basis.Read(), Q1D, D1D);
   auto D = Reshape(padata.Read(), Q1D, Q1D, NE);
   auto M = Reshape(eadata.ReadWrite(), D1D, D1D, D1D, D1D, NE);
   MFEM_FORALL_3D(e, NE, D1D, D1D, 1,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      double r_B[MQ1][MD1];
      for (int d = 0; d < D1D; d++)
      {
         for (int q = 0; q < Q1D; q++)
         {
            r_B[q][d] = B(q,d);
         }
      }
      MFEM_SHARED double s_D[MQ1][MQ1];
      MFEM_FOREACH_THREAD(k1,x,Q1D)
      {
         MFEM_FOREACH_THREAD(k2,y,Q1D)
         {
            s_D[k1][k2] = D(k1,k2,e);
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(i1,x,D1D)
      {
         MFEM_FOREACH_THREAD(i2,y,D1D)
         {
            for (int j1 = 0; j1 < D1D; ++j1)
            {
               for (int j2 = 0; j2 < D1D; ++j2)
               {
                  double val = 0.0;
                  for (int k1 = 0; k1 < Q1D; ++k1)
                  {
                     for (int k2 = 0; k2 < Q1D; ++k2)
                     {
                        val += r_B[k1][i1] * r_B[k1][j1]
                               * r_B[k2][i2] * r_B[k2][j2]
                               * s_D[k1][k2];
                     }
                  }
                  if (add)
                  {
                     M(i1, i2, j1, j2, e) += val;
                  }
                  else
                  {
                     M(i1, i2, j1, j2, e) = val;
                  }
               }
            }
         }
      }
   });
}

template<int T_D1D = 0, int T_Q1D = 0>
static void EAMassAssemble3D(const int NE,
                             const Array<double> &basis,
                             const Vector &padata,
                             Vector &eadata,
                             const bool add,
                             const int d1d = 0,
                             const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   auto B = Reshape(basis.Read(), Q1D, D1D);
   auto D = Reshape(padata.Read(), Q1D, Q1D, Q1D, NE);
   auto M = Reshape(eadata.ReadWrite(), D1D, D1D, D1D, D1D, D1D, D1D, NE);
   MFEM_FORALL_3D(e, NE, D1D, D1D, D1D,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      double r_B[MQ1][MD1];
      for (int d = 0; d < D1D; d++)
      {
         for (int q = 0; q < Q1D; q++)
         {
            r_B[q][d] = B(q,d);
         }
      }
      MFEM_SHARED double s_D[MQ1][MQ1][MQ1];
      MFEM_FOREACH_THREAD(k1,x,Q1D)
      {
         MFEM_FOREACH_THREAD(k2,y,Q1D)
         {
            MFEM_FOREACH_THREAD(k3,z,Q1D)
            {
               s_D[k1][k2][k3] = D(k1,k2,k3,e);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(i1,x,D1D)
      {
         MFEM_FOREACH_THREAD(i2,y,D1D)
         {
            MFEM_FOREACH_THREAD(i3,z,D1D)
            {
               for (int j1 = 0; j1 < D1D; ++j1)
               {
                  for (int j2 = 0; j2 < D1D; ++j2)
                  {
                     for (int j3 = 0; j3 < D1D; ++j3)
                     {
                        double val = 0.0;
                        for (int k1 = 0; k1 < Q1D; ++k1)
                        {
                           for (int k2 = 0; k2 < Q1D; ++k2)
                           {
                              for (int k3 = 0; k3 < Q1D; ++k3)
                              {
                                 val += r_B[k1][i1] * r_B[k1][j1]
                                        * r_B[k2][i2] * r_B[k2][j2]
                                        * r_B[k3][i3] * r_B[k3][j3]
                                        * s_D[k1][k2][k3];
                              }
                           }
                        }
                        if (add)
                        {
                           M(i1, i2, i3, j1, j2, j3, e) += val;
                        }
                        else
                        {
                           M(i1, i2, i3, j1, j2, j3, e) = val;
                        }
                     }
                  }
               }
            }
         }
      }
   });
}

}  // namespace mfem
