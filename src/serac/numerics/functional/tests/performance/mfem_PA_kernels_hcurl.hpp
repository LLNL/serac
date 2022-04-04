#include "mfem.hpp"
#include "mfem/general/forall.hpp"

namespace mfem {

template<int MAX_D1D = HCURL_MAX_D1D, int MAX_Q1D = HCURL_MAX_Q1D>
static void PACurlCurlApply3D(const int D1D,
                              const int Q1D,
                              const bool symmetric,
                              const int NE,
                              const Array<double> &bo,
                              const Array<double> &bc,
                              const Array<double> &bot,
                              const Array<double> &bct,
                              const Array<double> &gc,
                              const Array<double> &gct,
                              const Vector &pa_data,
                              const Vector &x,
                              Vector &y)
{
   MFEM_VERIFY(D1D <= MAX_D1D, "Error: D1D > MAX_D1D");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "Error: Q1D > MAX_Q1D");
   // Using (\nabla\times u) F = 1/det(dF) dF \hat{\nabla}\times\hat{u} (p. 78 of Monk), we get
   // (\nabla\times u) \cdot (\nabla\times v) = 1/det(dF)^2 \hat{\nabla}\times\hat{u}^T dF^T dF \hat{\nabla}\times\hat{v}
   // If c = 0, \hat{\nabla}\times\hat{u} reduces to [0, (u_0)_{x_2}, -(u_0)_{x_1}]
   // If c = 1, \hat{\nabla}\times\hat{u} reduces to [-(u_1)_{x_2}, 0, (u_1)_{x_0}]
   // If c = 2, \hat{\nabla}\times\hat{u} reduces to [(u_2)_{x_1}, -(u_2)_{x_0}, 0]

   constexpr static int VDIM = 3;

   auto Bo = Reshape(bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(bc.Read(), Q1D, D1D);
   auto Bot = Reshape(bot.Read(), D1D-1, Q1D);
   auto Bct = Reshape(bct.Read(), D1D, Q1D);
   auto Gc = Reshape(gc.Read(), Q1D, D1D);
   auto Gct = Reshape(gct.Read(), D1D, Q1D);
   auto op = Reshape(pa_data.Read(), Q1D, Q1D, Q1D, (symmetric ? 6 : 9), NE);
   auto X = Reshape(x.Read(), 3*(D1D-1)*D1D*D1D, NE);
   auto Y = Reshape(y.ReadWrite(), 3*(D1D-1)*D1D*D1D, NE);

   MFEM_FORALL(e, NE,
   {
      double curl[MAX_Q1D][MAX_Q1D][MAX_Q1D][VDIM];
      // curl[qz][qy][qx] will be computed as the vector curl at each quadrature point.

      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int c = 0; c < VDIM; ++c)
               {
                  curl[qz][qy][qx][c] = 0.0;
               }
            }
         }
      }

      // We treat x, y, z components separately for optimization specific to each.

      int osc = 0;

      {
         // x component
         const int D1Dz = D1D;
         const int D1Dy = D1D;
         const int D1Dx = D1D - 1;

         for (int dz = 0; dz < D1Dz; ++dz)
         {
            double gradXY[MAX_Q1D][MAX_Q1D][2];
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  for (int d = 0; d < 2; ++d)
                  {
                     gradXY[qy][qx][d] = 0.0;
                  }
               }
            }

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               double massX[MAX_Q1D];
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  massX[qx] = 0.0;
               }

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  const double t = X(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     massX[qx] += t * Bo(qx,dx);
                  }
               }

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const double wy = Bc(qy,dy);
                  const double wDy = Gc(qy,dy);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const double wx = massX[qx];
                     gradXY[qy][qx][0] += wx * wDy;
                     gradXY[qy][qx][1] += wx * wy;
                  }
               }
            }

            for (int qz = 0; qz < Q1D; ++qz)
            {
               const double wz = Bc(qz,dz);
               const double wDz = Gc(qz,dz);
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     // \hat{\nabla}\times\hat{u} is [0, (u_0)_{x_2}, -(u_0)_{x_1}]
                     curl[qz][qy][qx][1] += gradXY[qy][qx][1] * wDz; // (u_0)_{x_2}
                     curl[qz][qy][qx][2] -= gradXY[qy][qx][0] * wz;  // -(u_0)_{x_1}
                  }
               }
            }
         }

         osc += D1Dx * D1Dy * D1Dz;
      }

      {
         // y component
         const int D1Dz = D1D;
         const int D1Dy = D1D - 1;
         const int D1Dx = D1D;

         for (int dz = 0; dz < D1Dz; ++dz)
         {
            double gradXY[MAX_Q1D][MAX_Q1D][2];
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  for (int d = 0; d < 2; ++d)
                  {
                     gradXY[qy][qx][d] = 0.0;
                  }
               }
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               double massY[MAX_Q1D];
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  massY[qy] = 0.0;
               }

               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  const double t = X(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     massY[qy] += t * Bo(qy,dy);
                  }
               }

               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const double wx = Bc(qx,dx);
                  const double wDx = Gc(qx,dx);
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     const double wy = massY[qy];
                     gradXY[qy][qx][0] += wDx * wy;
                     gradXY[qy][qx][1] += wx * wy;
                  }
               }
            }

            for (int qz = 0; qz < Q1D; ++qz)
            {
               const double wz = Bc(qz,dz);
               const double wDz = Gc(qz,dz);
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     // \hat{\nabla}\times\hat{u} is [-(u_1)_{x_2}, 0, (u_1)_{x_0}]
                     curl[qz][qy][qx][0] -= gradXY[qy][qx][1] * wDz; // -(u_1)_{x_2}
                     curl[qz][qy][qx][2] += gradXY[qy][qx][0] * wz;  // (u_1)_{x_0}
                  }
               }
            }
         }

         osc += D1Dx * D1Dy * D1Dz;
      }

      {
         // z component
         const int D1Dz = D1D - 1;
         const int D1Dy = D1D;
         const int D1Dx = D1D;

         for (int dx = 0; dx < D1Dx; ++dx)
         {
            double gradYZ[MAX_Q1D][MAX_Q1D][2];
            for (int qz = 0; qz < Q1D; ++qz)
            {
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int d = 0; d < 2; ++d)
                  {
                     gradYZ[qz][qy][d] = 0.0;
                  }
               }
            }

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               double massZ[MAX_Q1D];
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  massZ[qz] = 0.0;
               }

               for (int dz = 0; dz < D1Dz; ++dz)
               {
                  const double t = X(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     massZ[qz] += t * Bo(qz,dz);
                  }
               }

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const double wy = Bc(qy,dy);
                  const double wDy = Gc(qy,dy);
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     const double wz = massZ[qz];
                     gradYZ[qz][qy][0] += wz * wy;
                     gradYZ[qz][qy][1] += wz * wDy;
                  }
               }
            }

            for (int qx = 0; qx < Q1D; ++qx)
            {
               const double wx = Bc(qx,dx);
               const double wDx = Gc(qx,dx);

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     // \hat{\nabla}\times\hat{u} is [(u_2)_{x_1}, -(u_2)_{x_0}, 0]
                     curl[qz][qy][qx][0] += gradYZ[qz][qy][1] * wx;  // (u_2)_{x_1}
                     curl[qz][qy][qx][1] -= gradYZ[qz][qy][0] * wDx; // -(u_2)_{x_0}
                  }
               }
            }
         }
      }

      // Apply D operator.
      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const double O11 = op(qx,qy,qz,0,e);
               const double O12 = op(qx,qy,qz,1,e);
               const double O13 = op(qx,qy,qz,2,e);
               const double O21 = symmetric ? O12 : op(qx,qy,qz,3,e);
               const double O22 = symmetric ? op(qx,qy,qz,3,e) : op(qx,qy,qz,4,e);
               const double O23 = symmetric ? op(qx,qy,qz,4,e) : op(qx,qy,qz,5,e);
               const double O31 = symmetric ? O13 : op(qx,qy,qz,6,e);
               const double O32 = symmetric ? O23 : op(qx,qy,qz,7,e);
               const double O33 = symmetric ? op(qx,qy,qz,5,e) : op(qx,qy,qz,8,e);

               const double c1 = (O11 * curl[qz][qy][qx][0]) + (O12 * curl[qz][qy][qx][1]) +
                                 (O13 * curl[qz][qy][qx][2]);
               const double c2 = (O21 * curl[qz][qy][qx][0]) + (O22 * curl[qz][qy][qx][1]) +
                                 (O23 * curl[qz][qy][qx][2]);
               const double c3 = (O31 * curl[qz][qy][qx][0]) + (O32 * curl[qz][qy][qx][1]) +
                                 (O33 * curl[qz][qy][qx][2]);

               curl[qz][qy][qx][0] = c1;
               curl[qz][qy][qx][1] = c2;
               curl[qz][qy][qx][2] = c3;
            }
         }
      }

      // x component
      osc = 0;
      {
         const int D1Dz = D1D;
         const int D1Dy = D1D;
         const int D1Dx = D1D - 1;

         for (int qz = 0; qz < Q1D; ++qz)
         {
            double gradXY12[MAX_D1D][MAX_D1D];
            double gradXY21[MAX_D1D][MAX_D1D];

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  gradXY12[dy][dx] = 0.0;
                  gradXY21[dy][dx] = 0.0;
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               double massX[MAX_D1D][2];
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  for (int n = 0; n < 2; ++n)
                  {
                     massX[dx][n] = 0.0;
                  }
               }
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     const double wx = Bot(dx,qx);

                     massX[dx][0] += wx * curl[qz][qy][qx][1];
                     massX[dx][1] += wx * curl[qz][qy][qx][2];
                  }
               }
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  const double wy = Bct(dy,qy);
                  const double wDy = Gct(dy,qy);

                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     gradXY21[dy][dx] += massX[dx][0] * wy;
                     gradXY12[dy][dx] += massX[dx][1] * wDy;
                  }
               }
            }

            for (int dz = 0; dz < D1Dz; ++dz)
            {
               const double wz = Bct(dz,qz);
               const double wDz = Gct(dz,qz);
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     // \hat{\nabla}\times\hat{u} is [0, (u_0)_{x_2}, -(u_0)_{x_1}]
                     // (u_0)_{x_2} * (op * curl)_1 - (u_0)_{x_1} * (op * curl)_2
                     Y(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc,
                       e) += (gradXY21[dy][dx] * wDz) - (gradXY12[dy][dx] * wz);
                  }
               }
            }
         }  // loop qz

         osc += D1Dx * D1Dy * D1Dz;
      }

      // y component
      {
         const int D1Dz = D1D;
         const int D1Dy = D1D - 1;
         const int D1Dx = D1D;

         for (int qz = 0; qz < Q1D; ++qz)
         {
            double gradXY02[MAX_D1D][MAX_D1D];
            double gradXY20[MAX_D1D][MAX_D1D];

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  gradXY02[dy][dx] = 0.0;
                  gradXY20[dy][dx] = 0.0;
               }
            }
            for (int qx = 0; qx < Q1D; ++qx)
            {
               double massY[MAX_D1D][2];
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  massY[dy][0] = 0.0;
                  massY[dy][1] = 0.0;
               }
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int dy = 0; dy < D1Dy; ++dy)
                  {
                     const double wy = Bot(dy,qy);

                     massY[dy][0] += wy * curl[qz][qy][qx][2];
                     massY[dy][1] += wy * curl[qz][qy][qx][0];
                  }
               }
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  const double wx = Bct(dx,qx);
                  const double wDx = Gct(dx,qx);

                  for (int dy = 0; dy < D1Dy; ++dy)
                  {
                     gradXY02[dy][dx] += massY[dy][0] * wDx;
                     gradXY20[dy][dx] += massY[dy][1] * wx;
                  }
               }
            }

            for (int dz = 0; dz < D1Dz; ++dz)
            {
               const double wz = Bct(dz,qz);
               const double wDz = Gct(dz,qz);
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     // \hat{\nabla}\times\hat{u} is [-(u_1)_{x_2}, 0, (u_1)_{x_0}]
                     // -(u_1)_{x_2} * (op * curl)_0 + (u_1)_{x_0} * (op * curl)_2
                     Y(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc,
                       e) += (-gradXY20[dy][dx] * wDz) + (gradXY02[dy][dx] * wz);
                  }
               }
            }
         }  // loop qz

         osc += D1Dx * D1Dy * D1Dz;
      }

      // z component
      {
         const int D1Dz = D1D - 1;
         const int D1Dy = D1D;
         const int D1Dx = D1D;

         for (int qx = 0; qx < Q1D; ++qx)
         {
            double gradYZ01[MAX_D1D][MAX_D1D];
            double gradYZ10[MAX_D1D][MAX_D1D];

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int dz = 0; dz < D1Dz; ++dz)
               {
                  gradYZ01[dz][dy] = 0.0;
                  gradYZ10[dz][dy] = 0.0;
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               double massZ[MAX_D1D][2];
               for (int dz = 0; dz < D1Dz; ++dz)
               {
                  for (int n = 0; n < 2; ++n)
                  {
                     massZ[dz][n] = 0.0;
                  }
               }
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  for (int dz = 0; dz < D1Dz; ++dz)
                  {
                     const double wz = Bot(dz,qz);

                     massZ[dz][0] += wz * curl[qz][qy][qx][0];
                     massZ[dz][1] += wz * curl[qz][qy][qx][1];
                  }
               }
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  const double wy = Bct(dy,qy);
                  const double wDy = Gct(dy,qy);

                  for (int dz = 0; dz < D1Dz; ++dz)
                  {
                     gradYZ01[dz][dy] += wy * massZ[dz][1];
                     gradYZ10[dz][dy] += wDy * massZ[dz][0];
                  }
               }
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               const double wx = Bct(dx,qx);
               const double wDx = Gct(dx,qx);

               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  for (int dz = 0; dz < D1Dz; ++dz)
                  {
                     // \hat{\nabla}\times\hat{u} is [(u_2)_{x_1}, -(u_2)_{x_0}, 0]
                     // (u_2)_{x_1} * (op * curl)_0 - (u_2)_{x_0} * (op * curl)_1
                     Y(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc,
                       e) += (gradYZ10[dz][dy] * wx) - (gradYZ01[dz][dy] * wDx);
                  }
               }
            }
         }  // loop qx
      }
   }); // end of element loop
}

template<int MAX_D1D = HCURL_MAX_D1D, int MAX_Q1D = HCURL_MAX_Q1D>
static void SmemPACurlCurlApply3D(const int D1D,
                                  const int Q1D,
                                  const bool symmetric,
                                  const int NE,
                                  const Array<double> &bo,
                                  const Array<double> &bc,
                                  const Array<double> &bot,
                                  const Array<double> &bct,
                                  const Array<double> &gc,
                                  const Array<double> &gct,
                                  const Vector &pa_data,
                                  const Vector &x,
                                  Vector &y)
{
   MFEM_VERIFY(D1D <= MAX_D1D, "Error: D1D > MAX_D1D");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "Error: Q1D > MAX_Q1D");
   // Using (\nabla\times u) F = 1/det(dF) dF \hat{\nabla}\times\hat{u} (p. 78 of Monk), we get
   // (\nabla\times u) \cdot (\nabla\times v) = 1/det(dF)^2 \hat{\nabla}\times\hat{u}^T dF^T dF \hat{\nabla}\times\hat{v}
   // If c = 0, \hat{\nabla}\times\hat{u} reduces to [0, (u_0)_{x_2}, -(u_0)_{x_1}]
   // If c = 1, \hat{\nabla}\times\hat{u} reduces to [-(u_1)_{x_2}, 0, (u_1)_{x_0}]
   // If c = 2, \hat{\nabla}\times\hat{u} reduces to [(u_2)_{x_1}, -(u_2)_{x_0}, 0]

   auto Bo = Reshape(bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(bc.Read(), Q1D, D1D);
   auto Gc = Reshape(gc.Read(), Q1D, D1D);
   auto op = Reshape(pa_data.Read(), Q1D, Q1D, Q1D, symmetric ? 6 : 9, NE);
   auto X = Reshape(x.Read(), 3*(D1D-1)*D1D*D1D, NE);
   auto Y = Reshape(y.ReadWrite(), 3*(D1D-1)*D1D*D1D, NE);

   const int s = symmetric ? 6 : 9;

   auto device_kernel = [=] MFEM_DEVICE (int e)
   {
      constexpr int VDIM = 3;

      MFEM_SHARED double sBo[MAX_D1D][MAX_Q1D];
      MFEM_SHARED double sBc[MAX_D1D][MAX_Q1D];
      MFEM_SHARED double sGc[MAX_D1D][MAX_Q1D];

      double ope[9];
      MFEM_SHARED double sop[9][MAX_Q1D][MAX_Q1D];
      MFEM_SHARED double curl[MAX_Q1D][MAX_Q1D][3];

      MFEM_SHARED double sX[MAX_D1D][MAX_D1D][MAX_D1D];

      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qz,z,Q1D)
            {
               for (int i=0; i<s; ++i)
               {
                  ope[i] = op(qx,qy,qz,i,e);
               }
            }
         }
      }

      const int tidx = MFEM_THREAD_ID(x);
      const int tidy = MFEM_THREAD_ID(y);
      const int tidz = MFEM_THREAD_ID(z);

      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(d,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               sBc[d][q] = Bc(q,d);
               sGc[d][q] = Gc(q,d);
               if (d < D1D-1)
               {
                  sBo[d][q] = Bo(q,d);
               }
            }
         }
      }
      MFEM_SYNC_THREAD;

      for (int qz=0; qz < Q1D; ++qz)
      {
         if (tidz == qz)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
               {
                  for (int i=0; i<3; ++i)
                  {
                     curl[qy][qx][i] = 0.0;
                  }
               }
            }
         }

         int osc = 0;
         for (int c = 0; c < VDIM; ++c)  // loop over x, y, z components
         {
            const int D1Dz = (c == 2) ? D1D - 1 : D1D;
            const int D1Dy = (c == 1) ? D1D - 1 : D1D;
            const int D1Dx = (c == 0) ? D1D - 1 : D1D;

            MFEM_FOREACH_THREAD(dz,z,D1Dz)
            {
               MFEM_FOREACH_THREAD(dy,y,D1Dy)
               {
                  MFEM_FOREACH_THREAD(dx,x,D1Dx)
                  {
                     sX[dz][dy][dx] = X(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  }
               }
            }
            MFEM_SYNC_THREAD;

            if (tidz == qz)
            {
               if (c == 0)
               {
                  for (int i=0; i<s; ++i)
                  {
                     sop[i][tidx][tidy] = ope[i];
                  }
               }

               MFEM_FOREACH_THREAD(qy,y,Q1D)
               {
                  MFEM_FOREACH_THREAD(qx,x,Q1D)
                  {
                     double u = 0.0;
                     double v = 0.0;

                     // We treat x, y, z components separately for optimization specific to each.
                     if (c == 0) // x component
                     {
                        // \hat{\nabla}\times\hat{u} is [0, (u_0)_{x_2}, -(u_0)_{x_1}]

                        for (int dz = 0; dz < D1Dz; ++dz)
                        {
                           const double wz = sBc[dz][qz];
                           const double wDz = sGc[dz][qz];

                           for (int dy = 0; dy < D1Dy; ++dy)
                           {
                              const double wy = sBc[dy][qy];
                              const double wDy = sGc[dy][qy];

                              for (int dx = 0; dx < D1Dx; ++dx)
                              {
                                 const double wx = sX[dz][dy][dx] * sBo[dx][qx];
                                 u += wx * wDy * wz;
                                 v += wx * wy * wDz;
                              }
                           }
                        }

                        curl[qy][qx][1] += v; // (u_0)_{x_2}
                        curl[qy][qx][2] -= u;  // -(u_0)_{x_1}
                     }
                     else if (c == 1)  // y component
                     {
                        // \hat{\nabla}\times\hat{u} is [-(u_1)_{x_2}, 0, (u_1)_{x_0}]

                        for (int dz = 0; dz < D1Dz; ++dz)
                        {
                           const double wz = sBc[dz][qz];
                           const double wDz = sGc[dz][qz];

                           for (int dy = 0; dy < D1Dy; ++dy)
                           {
                              const double wy = sBo[dy][qy];

                              for (int dx = 0; dx < D1Dx; ++dx)
                              {
                                 const double t = sX[dz][dy][dx];
                                 const double wx = t * sBc[dx][qx];
                                 const double wDx = t * sGc[dx][qx];

                                 u += wDx * wy * wz;
                                 v += wx * wy * wDz;
                              }
                           }
                        }

                        curl[qy][qx][0] -= v; // -(u_1)_{x_2}
                        curl[qy][qx][2] += u; // (u_1)_{x_0}
                     }
                     else // z component
                     {
                        // \hat{\nabla}\times\hat{u} is [(u_2)_{x_1}, -(u_2)_{x_0}, 0]

                        for (int dz = 0; dz < D1Dz; ++dz)
                        {
                           const double wz = sBo[dz][qz];

                           for (int dy = 0; dy < D1Dy; ++dy)
                           {
                              const double wy = sBc[dy][qy];
                              const double wDy = sGc[dy][qy];

                              for (int dx = 0; dx < D1Dx; ++dx)
                              {
                                 const double t = sX[dz][dy][dx];
                                 const double wx = t * sBc[dx][qx];
                                 const double wDx = t * sGc[dx][qx];

                                 u += wDx * wy * wz;
                                 v += wx * wDy * wz;
                              }
                           }
                        }

                        curl[qy][qx][0] += v; // (u_2)_{x_1}
                        curl[qy][qx][1] -= u; // -(u_2)_{x_0}
                     }
                  } // qx
               } // qy
            } // tidz == qz

            osc += D1Dx * D1Dy * D1Dz;
            MFEM_SYNC_THREAD;
         } // c

         double dxyz1 = 0.0;
         double dxyz2 = 0.0;
         double dxyz3 = 0.0;

         MFEM_FOREACH_THREAD(dz,z,D1D)
         {
            const double wcz = sBc[dz][qz];
            const double wcDz = sGc[dz][qz];
            const double wz = (dz < D1D-1) ? sBo[dz][qz] : 0.0;

            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               MFEM_FOREACH_THREAD(dx,x,D1D)
               {
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     const double wcy = sBc[dy][qy];
                     const double wcDy = sGc[dy][qy];
                     const double wy = (dy < D1D-1) ? sBo[dy][qy] : 0.0;

                     for (int qx = 0; qx < Q1D; ++qx)
                     {
                        const double O11 = sop[0][qx][qy];
                        const double O12 = sop[1][qx][qy];
                        const double O13 = sop[2][qx][qy];
                        const double O21 = symmetric ? O12 : sop[3][qx][qy];
                        const double O22 = symmetric ? sop[3][qx][qy] : sop[4][qx][qy];
                        const double O23 = symmetric ? sop[4][qx][qy] : sop[5][qx][qy];
                        const double O31 = symmetric ? O13 : sop[6][qx][qy];
                        const double O32 = symmetric ? O23 : sop[7][qx][qy];
                        const double O33 = symmetric ? sop[5][qx][qy] : sop[8][qx][qy];

                        const double c1 = (O11 * curl[qy][qx][0]) + (O12 * curl[qy][qx][1]) +
                                          (O13 * curl[qy][qx][2]);
                        const double c2 = (O21 * curl[qy][qx][0]) + (O22 * curl[qy][qx][1]) +
                                          (O23 * curl[qy][qx][2]);
                        const double c3 = (O31 * curl[qy][qx][0]) + (O32 * curl[qy][qx][1]) +
                                          (O33 * curl[qy][qx][2]);

                        const double wcx = sBc[dx][qx];
                        const double wDx = sGc[dx][qx];

                        if (dx < D1D-1)
                        {
                           // \hat{\nabla}\times\hat{u} is [0, (u_0)_{x_2}, -(u_0)_{x_1}]
                           // (u_0)_{x_2} * (op * curl)_1 - (u_0)_{x_1} * (op * curl)_2
                           const double wx = sBo[dx][qx];
                           dxyz1 += (wx * c2 * wcy * wcDz) - (wx * c3 * wcDy * wcz);
                        }

                        // \hat{\nabla}\times\hat{u} is [-(u_1)_{x_2}, 0, (u_1)_{x_0}]
                        // -(u_1)_{x_2} * (op * curl)_0 + (u_1)_{x_0} * (op * curl)_2
                        dxyz2 += (-wy * c1 * wcx * wcDz) + (wy * c3 * wDx * wcz);

                        // \hat{\nabla}\times\hat{u} is [(u_2)_{x_1}, -(u_2)_{x_0}, 0]
                        // (u_2)_{x_1} * (op * curl)_0 - (u_2)_{x_0} * (op * curl)_1
                        dxyz3 += (wcDy * wz * c1 * wcx) - (wcy * wz * c2 * wDx);
                     } // qx
                  } // qy
               } // dx
            } // dy
         } // dz

         MFEM_SYNC_THREAD;

         MFEM_FOREACH_THREAD(dz,z,D1D)
         {
            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               MFEM_FOREACH_THREAD(dx,x,D1D)
               {
                  if (dx < D1D-1)
                  {
                     Y(dx + ((dy + (dz * D1D)) * (D1D-1)), e) += dxyz1;
                  }
                  if (dy < D1D-1)
                  {
                     Y(dx + ((dy + (dz * (D1D-1))) * D1D) + ((D1D-1)*D1D*D1D), e) += dxyz2;
                  }
                  if (dz < D1D-1)
                  {
                     Y(dx + ((dy + (dz * D1D)) * D1D) + (2*(D1D-1)*D1D*D1D), e) += dxyz3;
                  }
               }
            }
         }
      } // qz
   }; // end of element loop

   auto host_kernel = [&] MFEM_LAMBDA (int)
   {
      MFEM_ABORT_KERNEL("This kernel should only be used on GPU.");
   };

   ForallWrap<3>(true, NE, device_kernel, host_kernel, Q1D, Q1D, Q1D);
}

}
