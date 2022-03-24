#include "mfem.hpp"
#include "mfem/general/forall.hpp"

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

}
