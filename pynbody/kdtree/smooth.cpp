#define NO_IMPORT_ARRAY

#include "smooth.h"
#include "kd.h"
#include <assert.h>
#include <functional>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

 

  


void initParticleList(SMX smx) {
  smx->result = std::make_unique<std::vector<npy_intp>>();
  smx->result->reserve(100000);
  // not so large that it's expensive to reserve.
  // Not so small that we constantly need to get more space.
}

PyObject *getReturnParticleList(SMX smx) {
  // make a numpy array from smx->result
  npy_intp dims[1] = {static_cast<npy_intp>(smx->result->size())};
  PyObject *numpy_result = PyArray_SimpleNew(1, dims, NPY_INTP);

  std::copy(smx->result->begin(), smx->result->end(),
            static_cast<long *>(
                PyArray_DATA(reinterpret_cast<PyArrayObject *>(numpy_result))));
  smx->result.reset(nullptr);

  return numpy_result;
}
 
void
smSmoothInitStep(SMX smx) {
}

template <typename T> void smDomainDecomposition(KDContext* kd, int nprocs) {

  // AP 31/8/2014 - Here is the domain decomposition for nProcs>1
  // In principle one should do better by localizing the
  // domains -- the current approach is a seriously naive decomposition.
  // This will result in more snake collisions than necessary.
  // However in practice, up to nCpu = 8, the scaling is looking
  // pretty linear anyway so I'm leaving that for future.

  npy_intp pi;

  if (nprocs > 0) {
    for (pi = 0; pi < kd->nActive; ++pi) {
      SETSMOOTH(T, pi, -(float)(1 + pi % nprocs));
    }
  }
}

void smInitPriorityQueue(SMX smx) {
  /*
   ** Initialize Priority Queue.
   */

  
  std::cerr << "TEST of STL-based PQ" << std::endl;
  PriorityQueue<double> myq(5, 20);
  std::vector<int> particle_number_test =     {5,   0,   1,   10,  2,   3,    4,    6,   7,    8,   11};
  std::vector<double> particle_distance_test = {0.1, 0.7, 0.3, 2.2, 0.5, 0.6,  0.05, 0.9, 10.2, 1.0, 1.1};

  // iterate over the two lists simultaneously
  for (auto i = 0; i < particle_number_test.size(); ++i) {
    myq.push(particle_distance_test[i],particle_number_test[i]);
    std::cerr << myq.top() << "; " << myq.size() << std::endl;
    for(auto j = 0; j < 10; ++ j) {
      std::cerr << (myq.contains(j) ?"*":"o");
    }
    std::cerr << std::endl;
  }

  while(!myq.empty()) {
    myq.pop();
    std::cerr << myq.top() << "; " << myq.size() << std::endl;
    for(auto j = 0; j < 10; ++ j) {
      std::cerr << (myq.contains(j) ?"*":"o");
    }
    std::cerr << std::endl;
  }


}

template <typename T> T cubicSpline(SMX smx, T r2) {
  // Cubic Spline Kernel
  T rs;
  rs = 2.0 - sqrt(r2);
  if NPY_UNLIKELY(rs < 0)
    rs = 0;
  else if (r2 < 1.0)
    rs = (1.0 - 0.75 * rs * r2);
  else
    rs = 0.25 * rs * rs * rs;
  return rs;
}

template <typename T> T Wendland_kernel(SMX smx, T r2, int nSmooth) {
  // Wendland Kernel
  T rs;
  // Dehnen & Aly 2012 correction (1-0.0454684 at Ns=64) /
  float Wzero = (21 / 16.) * (1 - 0.0294 * pow(nSmooth * 0.01, -0.977));
  if NPY_UNLIKELY(r2 > 4.0)
    rs = 0;
  else if NPY_UNLIKELY(r2 <= 0)
    rs = Wzero;
  else {
    T au = sqrt(r2 * 0.25);
    rs = 1 - au;
    rs = rs * rs;
    rs = rs * rs;
    rs = (21 / 16.) * rs * (1 + 4 * au);
  }
  if NPY_UNLIKELY(rs < 0 && !smx->warnings) {
    fprintf(stderr, "Internal consistency error\n");
    smx->warnings = true;
  }
  return rs;
}

template <typename T>
void smDensitySym(SMX smx, npy_intp pi, int nSmooth, bool Wendland) {
  T fNorm, ih2, r2, rs, ih;
  npy_intp i, pj;
  KDContext* kd = smx->kd;

  ih = 1.0 / GETSMOOTH(T, pi);
  ih2 = ih * ih;
  fNorm = 0.5 * M_1_PI * ih * ih2;

  for (i = 0; i < nSmooth; ++i) {
    pj = smx->pList[i];
    r2 = smx->fList[i] * ih2;
    if (Wendland) {
      rs = Wendland_kernel(smx, r2, nSmooth);
    } else {
      rs = cubicSpline(smx, r2);
    }
    rs *= fNorm;
    ACCUM<T>(kd->pNumpyDen, kd->particleOffsets[pi],
             rs * GET<T>(kd->pNumpyMass, kd->particleOffsets[pj]));
    ACCUM<T>(kd->pNumpyDen, kd->particleOffsets[pj],
             rs * GET<T>(kd->pNumpyMass, kd->particleOffsets[pi]));
  }
}

template <typename T>
void smDensity(SMX smx, npy_intp pi, int nSmooth, bool Wendland) {
  T fNorm, ih2, r2, rs, ih;
  npy_intp j, pj, pi_iord;
  KDContext* kd = smx->kd;

  pi_iord = kd->particleOffsets[pi];
  ih = 1.0 / GET<T>(kd->pNumpySmooth, pi_iord);
  ih2 = ih * ih;
  fNorm = M_1_PI * ih * ih2;
  SET<T>(kd->pNumpyDen, pi_iord, 0.0);
  for (j = 0; j < nSmooth; ++j) {
    pj = smx->pList[j];
    r2 = smx->fList[j] * ih2;
    if (Wendland) {
      rs = Wendland_kernel(smx, r2, nSmooth);
    } else {
      rs = cubicSpline(smx, r2);
    }
    rs *= fNorm;
    ACCUM<T>(kd->pNumpyDen, pi_iord,
             rs * GET<T>(kd->pNumpyMass, kd->particleOffsets[pj]));
  }
}

template <typename Tf, typename Tq>
void smMeanQty1D(SMX smx, npy_intp pi, int nSmooth, bool Wendland) {
  Tf fNorm, ih2, r2, rs, ih, mass, rho;
  npy_intp j, pj, pi_iord;
  KDContext* kd = smx->kd;

  pi_iord = kd->particleOffsets[pi];
  ih = 1.0 / GET<Tf>(kd->pNumpySmooth, pi_iord);
  ih2 = ih * ih;
  fNorm = M_1_PI * ih * ih2;

  SET<Tq>(kd->pNumpyQtySmoothed, pi_iord, 0.0);

  for (j = 0; j < nSmooth; ++j) {
    pj = smx->pList[j];
    r2 = smx->fList[j] * ih2;
    if (Wendland) {
      rs = Wendland_kernel(smx, r2, nSmooth);
    } else {
      rs = cubicSpline(smx, r2);
    }
    rs *= fNorm;
    mass = GET<Tf>(kd->pNumpyMass, kd->particleOffsets[pj]);
    rho = GET<Tf>(kd->pNumpyDen, kd->particleOffsets[pj]);
    ACCUM<Tq>(kd->pNumpyQtySmoothed, pi_iord,
              rs * mass * GET<Tq>(kd->pNumpyQty, kd->particleOffsets[pj]) / rho);
  }
}

template <typename Tf, typename Tq>
void smMeanQtyND(SMX smx, npy_intp pi, int nSmooth, bool Wendland) {
  Tf fNorm, ih2, r2, rs, ih, mass, rho;
  npy_intp j, k, pj, pi_iord;
  KDContext* kd = smx->kd;

  pi_iord = kd->particleOffsets[pi];
  ih = 1.0 / GET<Tf>(kd->pNumpySmooth, pi_iord);
  ih2 = ih * ih;
  fNorm = M_1_PI * ih * ih2;

  for (k = 0; k < 3; ++k)
    SET2<Tq>(kd->pNumpyQtySmoothed, pi_iord, k, 0.0);

  for (j = 0; j < nSmooth; ++j) {
    pj = smx->pList[j];
    r2 = smx->fList[j] * ih2;
    if (Wendland) {
      rs = Wendland_kernel(smx, r2, nSmooth);
    } else {
      rs = cubicSpline(smx, r2);
    }
    rs *= fNorm;
    mass = GET<Tf>(kd->pNumpyMass, kd->particleOffsets[pj]);
    rho = GET<Tf>(kd->pNumpyDen, kd->particleOffsets[pj]);
    for (k = 0; k < 3; ++k) {
      ACCUM2<Tq>(kd->pNumpyQtySmoothed, pi_iord, k,
                 rs * mass * GET2<Tq>(kd->pNumpyQty, kd->particleOffsets[pj], k) /
                     rho);
    }
  }
}

template <typename Tf> Tf cubicSpline_gradient(Tf q, Tf ih, Tf r, Tf ih2) {
  // Kernel gradient
  Tf rs;
  if (q < 1.0)
    rs = -3.0 * ih + 2.25 * r * ih2;
  else
    rs = -0.75 * (2 - q) * (2 - q) / r;

  return rs;
}

template <typename Tf> Tf Wendland_gradient(Tf q, Tf r) {
  // Kernel gradient
  Tf rs;
  if (r < 1e-24)
    r = 1e-24; // Fix to avoid dividing by zero in case r = 0.
  // For this case q = 0 and rs = 0 in any case, so we can savely set r to a
  // tiny value.
  if (q < 2.0)
    rs = -5.0 * q * (1.0 - 0.5 * q) * (1.0 - 0.5 * q) * (1.0 - 0.5 * q) / r;

  return rs;
}

template <typename Tf, typename Tq>
void smCurlQty(SMX smx, npy_intp pi, int nSmooth, bool Wendland) {
  Tf fNorm, ih2, r2, r, rs, q2, q, ih, mass, rho, dqty[3], qty_i[3];
  npy_intp j, k, pj, pi_iord, pj_iord;
  KDContext* kd = smx->kd;
  Tf curl[3], x, y, z, dx, dy, dz;

  pi_iord = kd->particleOffsets[pi];
  ih = 1.0 / GET<Tf>(kd->pNumpySmooth, pi_iord);
  ih2 = ih * ih;
  fNorm = M_1_PI * ih2 * ih2;

  for (k = 0; k < 3; ++k) {
    SET2<Tq>(kd->pNumpyQtySmoothed, pi_iord, k, 0.0);
    qty_i[k] = GET2<Tq>(kd->pNumpyQty, pi_iord, k);
  }

  x = GET2<Tf>(kd->pNumpyPos, pi_iord, 0);
  y = GET2<Tf>(kd->pNumpyPos, pi_iord, 1);
  z = GET2<Tf>(kd->pNumpyPos, pi_iord, 2);

  for (j = 0; j < nSmooth; ++j) {
    pj = smx->pList[j];
    pj_iord = kd->particleOffsets[pj];
    dx = x - GET2<Tf>(kd->pNumpyPos, pj_iord, 0);
    dy = y - GET2<Tf>(kd->pNumpyPos, pj_iord, 1);
    dz = z - GET2<Tf>(kd->pNumpyPos, pj_iord, 2);

    r2 = smx->fList[j];
    q2 = r2 * ih2;
    r = sqrt(r2);
    q = sqrt(q2);

    // Kernel gradient
    if (Wendland) {
      rs = Wendland_gradient(q, r);
    } else {
      rs = cubicSpline_gradient(q, ih, r, ih2);
    }

    rs *= fNorm;

    mass = GET<Tf>(kd->pNumpyMass, pj_iord);
    rho = GET<Tf>(kd->pNumpyDen, pj_iord);

    for (k = 0; k < 3; ++k)
      dqty[k] = GET2<Tq>(kd->pNumpyQty, pj_iord, k) - qty_i[k];

    curl[0] = dy * dqty[2] - dz * dqty[1];
    curl[1] = dz * dqty[0] - dx * dqty[2];
    curl[2] = dx * dqty[1] - dy * dqty[0];

    for (k = 0; k < 3; ++k) {
      ACCUM2<Tq>(kd->pNumpyQtySmoothed, pi_iord, k, rs * curl[k] * mass / rho);
    }
  }
}

template <typename Tf, typename Tq>
void smDivQty(SMX smx, npy_intp pi, int nSmooth, bool Wendland) {
  Tf fNorm, ih2, r2, r, rs, q2, q, ih, mass, rho, div, dqty[3], qty_i[3];
  npy_intp j, k, pj, pi_iord, pj_iord;
  KDContext* kd = smx->kd;
  Tf x, y, z, dx, dy, dz;

  pi_iord = kd->particleOffsets[pi];
  ih = 1.0 / GET<Tf>(kd->pNumpySmooth, pi_iord);
  ih2 = ih * ih;
  fNorm = M_1_PI * ih2 * ih2;

  SET<Tq>(kd->pNumpyQtySmoothed, pi_iord, 0.0);

  x = GET2<Tf>(kd->pNumpyPos, pi_iord, 0);
  y = GET2<Tf>(kd->pNumpyPos, pi_iord, 1);
  z = GET2<Tf>(kd->pNumpyPos, pi_iord, 2);

  for (k = 0; k < 3; ++k)
    qty_i[k] = GET2<Tq>(kd->pNumpyQty, pi_iord, k);

  for (j = 0; j < nSmooth; ++j) {
    pj = smx->pList[j];
    pj_iord = kd->particleOffsets[pj];
    dx = x - GET2<Tf>(kd->pNumpyPos, pj_iord, 0);
    dy = y - GET2<Tf>(kd->pNumpyPos, pj_iord, 1);
    dz = z - GET2<Tf>(kd->pNumpyPos, pj_iord, 2);

    r2 = smx->fList[j];
    q2 = r2 * ih2;
    r = sqrt(r2);
    q = sqrt(q2);
    // Kernel gradient
    if (Wendland) {
      rs = Wendland_gradient(q, r);
    } else {
      rs = cubicSpline_gradient(q, ih, r, ih2);
    }

    rs *= fNorm;

    mass = GET<Tf>(kd->pNumpyMass, pj_iord);
    rho = GET<Tf>(kd->pNumpyDen, pj_iord);

    for (k = 0; k < 3; ++k)
      dqty[k] = GET2<Tq>(kd->pNumpyQty, pj_iord, k) - qty_i[k];

    div = dx * dqty[0] + dy * dqty[1] + dz * dqty[2];

    ACCUM<Tq>(kd->pNumpyQtySmoothed, pi_iord, rs * div * mass / rho);
  }
}

template <typename Tf, typename Tq>
void smDispQtyND(SMX smx, npy_intp pi, int nSmooth, bool Wendland) {
  float fNorm, ih2, r2, rs, ih, mass, rho;
  npy_intp j, k, pj, pi_iord;
  KDContext* kd = smx->kd;
  float mean[3], tdiff;

  pi_iord = kd->particleOffsets[pi];
  ih = 1.0 / GET<Tf>(kd->pNumpySmooth, pi_iord);
  ih2 = ih * ih;
  fNorm = M_1_PI * ih * ih2;

  SET<Tq>(kd->pNumpyQtySmoothed, pi_iord, 0.0);

  for (k = 0; k < 3; ++k) {

    mean[k] = 0;
  }

  // pass 1: find mean

  for (j = 0; j < nSmooth; ++j) {
    pj = smx->pList[j];
    r2 = smx->fList[j] * ih2;
    if (Wendland) {
      rs = Wendland_kernel(smx, r2, nSmooth);
    } else {
      rs = cubicSpline(smx, r2);
    }
    rs *= fNorm;
    mass = GET<Tf>(kd->pNumpyMass, kd->particleOffsets[pj]);
    rho = GET<Tf>(kd->pNumpyDen, kd->particleOffsets[pj]);
    for (k = 0; k < 3; ++k)
      mean[k] += rs * mass * GET2<Tq>(kd->pNumpyQty, kd->particleOffsets[pj], k) / rho;
  }

  // pass 2: get variance

  for (j = 0; j < nSmooth; ++j) {
    pj = smx->pList[j];
    r2 = smx->fList[j] * ih2;
    if (Wendland) {
      rs = Wendland_kernel(smx, r2, nSmooth);
    } else {
      rs = cubicSpline(smx, r2);
    }
    rs *= fNorm;
    mass = GET<Tf>(kd->pNumpyMass, kd->particleOffsets[pj]);
    rho = GET<Tf>(kd->pNumpyDen, kd->particleOffsets[pj]);
    for (k = 0; k < 3; ++k) {
      tdiff = mean[k] - GET2<Tq>(kd->pNumpyQty, kd->particleOffsets[pj], k);
      ACCUM<Tq>(kd->pNumpyQtySmoothed, pi_iord,
                rs * mass * tdiff * tdiff / rho);
    }
  }

  // finally: take square root to get dispersion

  SET<Tq>(kd->pNumpyQtySmoothed, pi_iord,
          sqrt(GET<Tq>(kd->pNumpyQtySmoothed, pi_iord)));
}

template <typename Tf, typename Tq>
void smDispQty1D(SMX smx, npy_intp pi, int nSmooth, bool Wendland) {
  float fNorm, ih2, r2, rs, ih, mass, rho;
  npy_intp j, pj, pi_iord;
  KDContext* kd = smx->kd;
  Tq mean, tdiff;

  pi_iord = kd->particleOffsets[pi];
  ih = 1.0 / GET<Tf>(kd->pNumpySmooth, pi_iord);
  ih2 = ih * ih;
  fNorm = M_1_PI * ih * ih2;

  SET<Tq>(kd->pNumpyQtySmoothed, pi_iord, 0.0);

  mean = 0;

  // pass 1: find mean

  for (j = 0; j < nSmooth; ++j) {
    pj = smx->pList[j];
    r2 = smx->fList[j] * ih2;
    if (Wendland) {
      rs = Wendland_kernel(smx, r2, nSmooth);
    } else {
      rs = cubicSpline(smx, r2);
    }

    rs *= fNorm;
    mass = GET<Tf>(kd->pNumpyMass, kd->particleOffsets[pj]);
    rho = GET<Tf>(kd->pNumpyDen, kd->particleOffsets[pj]);
    mean += rs * mass * GET<Tq>(kd->pNumpyQty, kd->particleOffsets[pj]) / rho;
  }

  // pass 2: get variance

  for (j = 0; j < nSmooth; ++j) {
    pj = smx->pList[j];
    r2 = smx->fList[j] * ih2;
    if (Wendland) {
      rs = Wendland_kernel(smx, r2, nSmooth);
    } else {
      rs = cubicSpline(smx, r2);
    }
    rs *= fNorm;
    mass = GET<Tf>(kd->pNumpyMass, kd->particleOffsets[pj]);
    rho = GET<Tf>(kd->pNumpyDen, kd->particleOffsets[pj]);
    tdiff = mean - GET<Tq>(kd->pNumpyQty, kd->particleOffsets[pj]);
    ACCUM<Tq>(kd->pNumpyQtySmoothed, pi_iord, rs * mass * tdiff * tdiff / rho);
  }

  // finally: take square root to get dispersion

  SET<Tq>(kd->pNumpyQtySmoothed, pi_iord,
          sqrt(GET<Tq>(kd->pNumpyQtySmoothed, pi_iord)));
}

// instantiate the actual functions that are available:

// template void smBallSearch<double>(SMX smx, double fBall2, double *ri);

template void smDomainDecomposition<double>(KDContext* kd, int nprocs);

template npy_intp smSmoothStep<double>(SMX smx, int procid);

template void smDensitySym<double>(SMX smx, npy_intp pi, int nSmooth,
                                   
                                   bool Wendland);

template void smDensity<double>(SMX smx, npy_intp pi, int nSmooth,
                                 bool Wendland);

// template void smBallSearch<float>(SMX smx, double fBall2, double *ri);

template void smDomainDecomposition<float>(KDContext* kd, int nprocs);

template npy_intp smSmoothStep<float>(SMX smx, int procid);

template void smDensitySym<float>(SMX smx, npy_intp pi, int nSmooth,
                                   bool Wendland);

template void smDensity<float>(SMX smx, npy_intp pi, int nSmooth,
                                bool Wendland);

template void smMeanQty1D<double, double>(SMX smx, npy_intp pi, int nSmooth,
                                          
                                          bool Wendland);

template void smMeanQtyND<double, double>(SMX smx, npy_intp pi, int nSmooth,
                                          
                                          bool Wendland);

template void smDispQty1D<double, double>(SMX smx, npy_intp pi, int nSmooth,
                                          
                                          bool Wendland);

template void smDispQtyND<double, double>(SMX smx, npy_intp pi, int nSmooth,
                                          
                                          bool Wendland);

template void smCurlQty<double, double>(SMX smx, npy_intp pi, int nSmooth,
                                        
                                        bool Wendland);

template void smDivQty<double, double>(SMX smx, npy_intp pi, int nSmooth,
                                       
                                       bool Wendland);

template void smMeanQty1D<double, float>(SMX smx, npy_intp pi, int nSmooth,
                                         
                                         bool Wendland);

template void smMeanQtyND<double, float>(SMX smx, npy_intp pi, int nSmooth,
                                         
                                         bool Wendland);

template void smDispQty1D<double, float>(SMX smx, npy_intp pi, int nSmooth,
                                         
                                         bool Wendland);

template void smDispQtyND<double, float>(SMX smx, npy_intp pi, int nSmooth,
                                         
                                         bool Wendland);

template void smCurlQty<double, float>(SMX smx, npy_intp pi, int nSmooth,
                                       
                                       bool Wendland);

template void smDivQty<double, float>(SMX smx, npy_intp pi, int nSmooth,
                                      
                                      bool Wendland);

template void smMeanQty1D<float, double>(SMX smx, npy_intp pi, int nSmooth,
                                         
                                         bool Wendland);

template void smMeanQtyND<float, double>(SMX smx, npy_intp pi, int nSmooth,
                                         
                                         bool Wendland);

template void smDispQty1D<float, double>(SMX smx, npy_intp pi, int nSmooth,
                                         
                                         bool Wendland);

template void smDispQtyND<float, double>(SMX smx, npy_intp pi, int nSmooth,
                                         
                                         bool Wendland);

template void smCurlQty<float, double>(SMX smx, npy_intp pi, int nSmooth,
                                       
                                       bool Wendland);

template void smDivQty<float, double>(SMX smx, npy_intp pi, int nSmooth,
                                      
                                      bool Wendland);

template void smMeanQty1D<float, float>(SMX smx, npy_intp pi, int nSmooth,
                                        
                                        bool Wendland);

template void smMeanQtyND<float, float>(SMX smx, npy_intp pi, int nSmooth,
                                        
                                        bool Wendland);

template void smDispQty1D<float, float>(SMX smx, npy_intp pi, int nSmooth,
                                        
                                        bool Wendland);

template void smDispQtyND<float, float>(SMX smx, npy_intp pi, int nSmooth,
                                        
                                        bool Wendland);

template void smCurlQty<float, float>(SMX smx, npy_intp pi, int nSmooth,
                                      
                                      bool Wendland);

template void smDivQty<float, float>(SMX smx, npy_intp pi, int nSmooth,
                                     
                                     bool Wendland);
