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

bool smCheckFits(KDContext* kd, float *fPeriod) {
  KDNode *root;
  npy_intp j;

  assert(kd->kdNodes != nullptr);
  root = &kd->kdNodes[ROOT];
  assert(root != nullptr);
  /*
   ** Check to make sure that the bounds of the simulation agree
   ** with the period specified, if not cause an error.
   */
  for (j = 0; j < 3; ++j) {
    if (root->bnd.fMax[j] - root->bnd.fMin[j] > fPeriod[j]) {
      return false;
    }
  }
  return true;
}

int smInit(SMX *psmx, KDContext* kd, int nSmooth, float *fPeriod) {
  SMX smx;
  KDNode *root;
  int j;
  int bError = 0;

  root = &kd->kdNodes[ROOT];
  assert(root != NULL);
  /*
   ** Check to make sure that the bounds of the simulation agree
   ** with the period specified, if not cause an error.
   */
  for (j = 0; j < 3; ++j) {
    if (root->bnd.fMax[j] - root->bnd.fMin[j] > fPeriod[j]) {
      PyErr_SetString(
          PyExc_ValueError,
          "The particles span a region larger than the specified boxsize");
      bError = 1;
    }
  }

  assert(nSmooth <= kd->nActive);
  smx = new smContext;
  smx->kd = kd;
  smx->nSmooth = nSmooth;
  smx->pq = (PQ *)malloc(nSmooth * sizeof(PQ));
  assert(smx->pq != NULL);
  PQ_INIT(smx->pq, nSmooth);
  smx->iMark = (char *)malloc(kd->nActive * sizeof(char));
  assert(smx->iMark != NULL);
  smx->nListSize = smx->nSmooth + RESMOOTH_SAFE;
  smx->fList = (float *)malloc(smx->nListSize * sizeof(float));
  assert(smx->fList != NULL);
  smx->pList = (npy_intp *)malloc(smx->nListSize * sizeof(npy_intp));
  assert(smx->pList != NULL);
  for (j = 0; j < 3; ++j)
    smx->fPeriod[j] = fPeriod[j];

  smx->nCurrent = 0;
  smx->pMutex = std::make_shared<std::mutex>();
  smx->smx_global = NULL;

  *psmx = smx;
  return (1);
}

SMX smInitThreadLocalCopy(SMX from) {

  SMX smx;
  KDNode *root;
  npy_intp pi;
  int j;

  root = &from->kd->kdNodes[ROOT];

  smx = new smContext;
  smx->kd = from->kd;
  smx->nSmooth = from->nSmooth;
  smx->pq = (PQ *)malloc(from->nSmooth * sizeof(PQ));
  assert(smx->pq != NULL);
  PQ_INIT(smx->pq, from->nSmooth);
  smx->iMark = (char *)malloc(from->kd->nActive * sizeof(char));
  assert(smx->iMark != NULL);
  smx->nListSize = from->nListSize;
  smx->fList = (float *)malloc(smx->nListSize * sizeof(float));
  assert(smx->fList != NULL);
  smx->pList = (npy_intp *)malloc(smx->nListSize * sizeof(npy_intp));
  assert(smx->pList != NULL);
  for (j = 0; j < 3; ++j)
    smx->fPeriod[j] = from->fPeriod[j];
  for (pi = 0; pi < smx->kd->nActive; ++pi) {
    smx->iMark[pi] = 0;
  }
  smx->pMutex = from->pMutex;

  smx->smx_global = from;
  smx->nCurrent = 0;

  smInitPriorityQueue(smx);
  return smx;
}

void smFinishThreadLocalCopy(SMX smx) {
  free(smx->pq);
  free(smx->fList);
  free(smx->pList);
  free(smx->iMark);
  delete smx;
}

#define WORKUNIT 1000

npy_intp smGetNext(SMX smx_local) {

  // synchronize warning state
  if (smx_local->warnings)
    smx_local->smx_global->warnings = true;

  npy_intp i = smx_local->nCurrent;

  if (smx_local->nCurrent % WORKUNIT == 0) {
    // we have reached the end of a work unit. Get and increment the global
    // counter.
    smx_local->pMutex->lock();
    smx_local->nCurrent = smx_local->smx_global->nCurrent;
    i = smx_local->nCurrent;
    smx_local->smx_global->nCurrent += WORKUNIT;
    smx_local->pMutex->unlock();
  }

  // i now has the next thing to be processed
  // increment the local counter

  smx_local->nCurrent += 1;

  return i;
}

void smFinish(SMX smx) {
  free(smx->iMark);
  free(smx->pq);
  free(smx->fList);
  free(smx->pList);

  delete smx;
}

template <typename T> void smBallSearch(SMX smx, float fBall2, float *ri) {
  KDNode *c;
  npy_intp *p;
  KDContext* kd;
  npy_intp cell, cp, ct, pj;
  T fDist2, dx, dy, dz, lx, ly, lz, sx, sy, sz, x, y, z;
  PQ *pq;

  kd = smx->kd;
  c = smx->kd->kdNodes;
  p = smx->kd->particleOffsets;
  pq = smx->pqHead;
  x = ri[0];
  y = ri[1];
  z = ri[2];
  lx = smx->fPeriod[0];
  ly = smx->fPeriod[1];
  lz = smx->fPeriod[2];
  cell = ROOT;
  /*
   ** First find the "local" Bucket.
   ** This could mearly be the closest bucket to ri[3].
   */
  while (cell < smx->kd->nSplit) {
    if (ri[c[cell].iDim] < c[cell].fSplit)
      cell = LOWER(cell);
    else
      cell = UPPER(cell);
  }
  /*
   ** Now start the search from the bucket given by cell!
   */
  for (pj = c[cell].pLower; pj <= c[cell].pUpper; ++pj) {
    dx = x - GET2<T>(kd->pNumpyPos, p[pj], 0);
    dy = y - GET2<T>(kd->pNumpyPos, p[pj], 1);
    dz = z - GET2<T>(kd->pNumpyPos, p[pj], 2);
    fDist2 = dx * dx + dy * dy + dz * dz;
    if (fDist2 < fBall2) {
      if (smx->iMark[pj])
        continue;
      smx->iMark[pq->p] = 0;
      smx->iMark[pj] = 1;
      pq->fKey = fDist2;
      pq->p = pj;
      pq->ax = 0.0;
      pq->ay = 0.0;
      pq->az = 0.0;
      PQ_REPLACE(pq);
      fBall2 = pq->fKey;
    }
  }
  while (cell != ROOT) {
    cp = SIBLING(cell);
    ct = cp;
    SETNEXT(ct, ROOT);
    while (1) {
      INTERSECT(c, cp, fBall2, lx, ly, lz, x, y, z, sx, sy, sz);
      /*
       ** We have an intersection to test.
       */
      if (cp < smx->kd->nSplit) {
        cp = LOWER(cp);
        continue;
      } else {
        for (pj = c[cp].pLower; pj <= c[cp].pUpper; ++pj) {
          dx = sx - GET2<T>(kd->pNumpyPos, p[pj], 0);
          dy = sy - GET2<T>(kd->pNumpyPos, p[pj], 1);
          dz = sz - GET2<T>(kd->pNumpyPos, p[pj], 2);
          fDist2 = dx * dx + dy * dy + dz * dz;
          if (fDist2 < fBall2) {
            if (smx->iMark[pj])
              continue;
            smx->iMark[pq->p] = 0;
            smx->iMark[pj] = 1;
            pq->fKey = fDist2;
            pq->p = pj;
            pq->ax = sx - x;
            pq->ay = sy - y;
            pq->az = sz - z;
            PQ_REPLACE(pq);
            fBall2 = pq->fKey;
          }
        }
      }
    GetNextCell:
      SETNEXT(cp, ROOT);
      if (cp == ct)
        break;
    }
    cell = PARENT(cell);
  }
  smx->pqHead = pq;
}

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

void smSmoothInitStep(SMX smx, int nProcs_for_smooth) {

  npy_intp pi;
  KDContext* kd = smx->kd;

  for (pi = 0; pi < kd->nActive; ++pi) {
    smx->iMark[pi] = 0;
  }

  smInitPriorityQueue(smx);
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

  PQ *pq, *pqLast;
  npy_intp pin, pj, pNext;
  float ax, ay, az;

  pqLast = &smx->pq[smx->nSmooth - 1];
  pin = 0;
  pNext = 1;
  ax = 0.0;
  ay = 0.0;
  az = 0.0;

  for (pq = smx->pq, pj = 0; pq <= pqLast; ++pq, ++pj) {
    smx->iMark[pj] = 1;
    pq->p = pj;
    pq->ax = ax;
    pq->ay = ay;
    pq->az = az;
  }
  smx->pin = pin;
  smx->pNext = pNext;
  smx->ax = ax;
  smx->ay = ay;
  smx->az = az;
}

template <typename T> npy_intp smSmoothStep(SMX smx, int procid) {
  KDNode *c;
  npy_intp *p;
  PQ *pq, *pqLast;
  KDContext* kd = smx->kd;
  npy_intp cell;
  npy_intp pi, pin, pj, pNext, nCnt, nSmooth;
  npy_intp nScanned = 0;

  float dx, dy, dz, x, y, z, h2, ax, ay, az;
  float proc_signal = -(float)(procid)-1.0;
  float ri[3];

  c = smx->kd->kdNodes;
  p = smx->kd->particleOffsets;
  pqLast = &smx->pq[smx->nSmooth - 1];
  nSmooth = smx->nSmooth;
  pin = smx->pin;
  pNext = smx->pNext;
  ax = smx->ax;
  ay = smx->ay;
  az = smx->az;

  if (GETSMOOTH(T, pin) >= 0) {
    // the first particle we are supposed to smooth is
    // actually already done. We need to search for another
    // suitable candidate. Preferably a long way away from other
    // threads, if this is threaded.

    if (pNext >= smx->kd->nActive)
      pNext = 0;

    while (GETSMOOTH(T, pNext) != proc_signal) {
      ++pNext;
      ++nScanned;
      if (pNext >= smx->kd->nActive)
        pNext = 0;
      if (nScanned == smx->kd->nActive) {
        // Nothing remains to be done.
        return -1;
      }
    }

    // mark the particle as 'processed' by assigning a dummy positive value
    // N.B. a race condition here doesn't matter since duplicating a bit of
    // work is more efficient than using a mutex (verified).
    SETSMOOTH(T, pNext, 10);

    pi = pNext;
    ++pNext;
    x = GET2<T>(kd->pNumpyPos, p[pi], 0);
    y = GET2<T>(kd->pNumpyPos, p[pi], 1);
    z = GET2<T>(kd->pNumpyPos, p[pi], 2);
    /*
    ** First find the "local" Bucket.
    ** This could merely be the closest bucket to ri[3].
    */
    cell = ROOT;
    while (cell < smx->kd->nSplit) {
      if (GET2<T>(kd->pNumpyPos, p[pi], c[cell].iDim) < c[cell].fSplit)
        cell = LOWER(cell);
      else
        cell = UPPER(cell);
    }
    /*
    ** Remove everything from the queue.
    */
    smx->pqHead = NULL;
    for (pq = smx->pq; pq <= pqLast; ++pq)
      smx->iMark[pq->p] = 0;
    /*
    ** Add everything from pj up to and including pj+nSmooth-1.
    */
    pj = c[cell].pLower;
    if (pj > smx->kd->nActive - nSmooth)
      pj = smx->kd->nActive - nSmooth;
    for (pq = smx->pq; pq <= pqLast; ++pq) {
      smx->iMark[pj] = 1;
      dx = x - GET2<T>(kd->pNumpyPos, p[pj], 0);
      dy = y - GET2<T>(kd->pNumpyPos, p[pj], 1);
      dz = z - GET2<T>(kd->pNumpyPos, p[pj], 2);
      pq->fKey = dx * dx + dy * dy + dz * dz;
      pq->p = pj++;
      pq->ax = 0.0;
      pq->ay = 0.0;
      pq->az = 0.0;
    }
    PQ_BUILD(smx->pq, nSmooth, smx->pqHead);
  } else {
    // Calculate priority queue using existing particles
    pi = pin;

    // Mark - see comment above
    SETSMOOTH(T, pi, 10);

    x = GET2<T>(kd->pNumpyPos, p[pi], 0);
    y = GET2<T>(kd->pNumpyPos, p[pi], 1);
    z = GET2<T>(kd->pNumpyPos, p[pi], 2);

    smx->pqHead = NULL;
    for (pq = smx->pq; pq <= pqLast; ++pq) {
      pq->ax -= ax;
      pq->ay -= ay;
      pq->az -= az;
      dx = x + pq->ax - GET2<T>(kd->pNumpyPos, p[pq->p], 0);
      dy = y + pq->ay - GET2<T>(kd->pNumpyPos, p[pq->p], 1);
      dz = z + pq->az - GET2<T>(kd->pNumpyPos, p[pq->p], 2);
      pq->fKey = dx * dx + dy * dy + dz * dz;
    }
    PQ_BUILD(smx->pq, nSmooth, smx->pqHead);
    ax = 0.0;
    ay = 0.0;
    az = 0.0;
  }

  for (int j = 0; j < 3; ++j) {
    ri[j] = GET2<T>(kd->pNumpyPos, p[pi], j);
  }

  smBallSearch<T>(smx, smx->pqHead->fKey, ri);
  SETSMOOTH(T, pi, 0.5 * sqrt(smx->pqHead->fKey));

  // p[pi].fSmooth = 0.5*sqrt(smx->pfBall2[pi]);
  /*
  ** Pick next particle, 'pin'.
  ** Create fList and pList for function 'fncSmooth'.
  */
  pin = pi;
  nCnt = 0;
  h2 = smx->pqHead->fKey;
  for (pq = smx->pq; pq <= pqLast; ++pq) {

    /* the next line is commented out because it results in the furthest
    particle being excluded from the nearest-neighbor list - this means
    that although the user requests 32 NN, she would get back 31. By
    including the furthest particle, the neighbor list is always 32 long */

    // if (pq == smx->pqHead) continue;
    if (nCnt >= smx->nListSize) {
      // no room left
      if (!smx->warnings)
        fprintf(stderr, "Smooth - particle cache too small for local density - "
                        "results will be incorrect\n");
      smx->warnings = true;
      break;
    }

    smx->pList[nCnt] = pq->p;
    smx->fList[nCnt++] = pq->fKey;

    if (GETSMOOTH(T, pq->p) >= 0)
      continue; // already done, don't re-do

    if (pq->fKey < h2) {
      pin = pq->p;
      h2 = pq->fKey;
      ax = pq->ax;
      ay = pq->ay;
      az = pq->az;
    }
  }

  smx->pi = pi;
  smx->pin = pin;
  smx->pNext = pNext;
  smx->ax = ax;
  smx->ay = ay;
  smx->az = az;

  return nCnt;
}

template <typename T> T cubicSpline(SMX smx, T r2) {
  // Cubic Spline Kernel
  T rs;
  rs = 2.0 - sqrt(r2);
  if (r2 < 1.0)
    rs = (1.0 - 0.75 * rs * r2);
  else
    rs = 0.25 * rs * rs * rs;
  if (rs < 0)
    rs = 0;
  return rs;
}

template <typename T> T Wendland_kernel(SMX smx, T r2, int nSmooth) {
  // Wendland Kernel
  T rs;
  // Dehnen & Aly 2012 correction (1-0.0454684 at Ns=64) /
  float Wzero = (21 / 16.) * (1 - 0.0294 * pow(nSmooth * 0.01, -0.977));
  if (r2 <= 0)
    rs = Wzero;
  else {
    T au = sqrt(r2 * 0.25);
    rs = 1 - au;
    rs = rs * rs;
    rs = rs * rs;
    rs = (21 / 16.) * rs * (1 + 4 * au);
  }
  if (rs < 0 && !smx->warnings) {
    fprintf(stderr, "Internal consistency error\n");
    smx->warnings = true;
  }
  return rs;
}

template <typename T>
void smDensitySym(SMX smx, npy_intp pi, int nSmooth, npy_intp *pList,
                  float *fList, bool Wendland) {
  T fNorm, ih2, r2, rs, ih;
  npy_intp i, pj;
  KDContext* kd = smx->kd;

  ih = 1.0 / GETSMOOTH(T, pi);
  ih2 = ih * ih;
  fNorm = 0.5 * M_1_PI * ih * ih2;

  for (i = 0; i < nSmooth; ++i) {
    pj = pList[i];
    r2 = fList[i] * ih2;
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
void smDensity(SMX smx, npy_intp pi, int nSmooth, npy_intp *pList, float *fList,
               bool Wendland) {
  T fNorm, ih2, r2, rs, ih;
  npy_intp j, pj, pi_iord;
  KDContext* kd = smx->kd;

  pi_iord = kd->particleOffsets[pi];
  ih = 1.0 / GET<T>(kd->pNumpySmooth, pi_iord);
  ih2 = ih * ih;
  fNorm = M_1_PI * ih * ih2;
  SET<T>(kd->pNumpyDen, pi_iord, 0.0);
  for (j = 0; j < nSmooth; ++j) {
    pj = pList[j];
    r2 = fList[j] * ih2;
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
void smMeanQty1D(SMX smx, npy_intp pi, int nSmooth, npy_intp *pList,
                 float *fList, bool Wendland) {
  Tf fNorm, ih2, r2, rs, ih, mass, rho;
  npy_intp j, pj, pi_iord;
  KDContext* kd = smx->kd;

  pi_iord = kd->particleOffsets[pi];
  ih = 1.0 / GET<Tf>(kd->pNumpySmooth, pi_iord);
  ih2 = ih * ih;
  fNorm = M_1_PI * ih * ih2;

  SET<Tq>(kd->pNumpyQtySmoothed, pi_iord, 0.0);

  for (j = 0; j < nSmooth; ++j) {
    pj = pList[j];
    r2 = fList[j] * ih2;
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
void smMeanQtyND(SMX smx, npy_intp pi, int nSmooth, npy_intp *pList,
                 float *fList, bool Wendland) {
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
    pj = pList[j];
    r2 = fList[j] * ih2;
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
void smCurlQty(SMX smx, npy_intp pi, int nSmooth, npy_intp *pList, float *fList,
               bool Wendland) {
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
    pj = pList[j];
    pj_iord = kd->particleOffsets[pj];
    dx = x - GET2<Tf>(kd->pNumpyPos, pj_iord, 0);
    dy = y - GET2<Tf>(kd->pNumpyPos, pj_iord, 1);
    dz = z - GET2<Tf>(kd->pNumpyPos, pj_iord, 2);

    r2 = fList[j];
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
void smDivQty(SMX smx, npy_intp pi, int nSmooth, npy_intp *pList, float *fList,
              bool Wendland) {
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
    pj = pList[j];
    pj_iord = kd->particleOffsets[pj];
    dx = x - GET2<Tf>(kd->pNumpyPos, pj_iord, 0);
    dy = y - GET2<Tf>(kd->pNumpyPos, pj_iord, 1);
    dz = z - GET2<Tf>(kd->pNumpyPos, pj_iord, 2);

    r2 = fList[j];
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
void smDispQtyND(SMX smx, npy_intp pi, int nSmooth, npy_intp *pList,
                 float *fList, bool Wendland) {
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
    pj = pList[j];
    r2 = fList[j] * ih2;
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
    pj = pList[j];
    r2 = fList[j] * ih2;
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
void smDispQty1D(SMX smx, npy_intp pi, int nSmooth, npy_intp *pList,
                 float *fList, bool Wendland) {
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
    pj = pList[j];
    r2 = fList[j] * ih2;
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
    pj = pList[j];
    r2 = fList[j] * ih2;
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

template void smBallSearch<double>(SMX smx, float fBall2, float *ri);

template void smDomainDecomposition<double>(KDContext* kd, int nprocs);

template npy_intp smSmoothStep<double>(SMX smx, int procid);

template void smDensitySym<double>(SMX smx, npy_intp pi, int nSmooth,
                                   npy_intp *pList, float *fList,
                                   bool Wendland);

template void smDensity<double>(SMX smx, npy_intp pi, int nSmooth,
                                npy_intp *pList, float *fList, bool Wendland);

template void smBallSearch<float>(SMX smx, float fBall2, float *ri);

template void smDomainDecomposition<float>(KDContext* kd, int nprocs);

template npy_intp smSmoothStep<float>(SMX smx, int procid);

template void smDensitySym<float>(SMX smx, npy_intp pi, int nSmooth,
                                  npy_intp *pList, float *fList, bool Wendland);

template void smDensity<float>(SMX smx, npy_intp pi, int nSmooth,
                               npy_intp *pList, float *fList, bool Wendland);

template void smMeanQty1D<double, double>(SMX smx, npy_intp pi, int nSmooth,
                                          npy_intp *pList, float *fList,
                                          bool Wendland);

template void smMeanQtyND<double, double>(SMX smx, npy_intp pi, int nSmooth,
                                          npy_intp *pList, float *fList,
                                          bool Wendland);

template void smDispQty1D<double, double>(SMX smx, npy_intp pi, int nSmooth,
                                          npy_intp *pList, float *fList,
                                          bool Wendland);

template void smDispQtyND<double, double>(SMX smx, npy_intp pi, int nSmooth,
                                          npy_intp *pList, float *fList,
                                          bool Wendland);

template void smCurlQty<double, double>(SMX smx, npy_intp pi, int nSmooth,
                                        npy_intp *pList, float *fList,
                                        bool Wendland);

template void smDivQty<double, double>(SMX smx, npy_intp pi, int nSmooth,
                                       npy_intp *pList, float *fList,
                                       bool Wendland);

template void smMeanQty1D<double, float>(SMX smx, npy_intp pi, int nSmooth,
                                         npy_intp *pList, float *fList,
                                         bool Wendland);

template void smMeanQtyND<double, float>(SMX smx, npy_intp pi, int nSmooth,
                                         npy_intp *pList, float *fList,
                                         bool Wendland);

template void smDispQty1D<double, float>(SMX smx, npy_intp pi, int nSmooth,
                                         npy_intp *pList, float *fList,
                                         bool Wendland);

template void smDispQtyND<double, float>(SMX smx, npy_intp pi, int nSmooth,
                                         npy_intp *pList, float *fList,
                                         bool Wendland);

template void smCurlQty<double, float>(SMX smx, npy_intp pi, int nSmooth,
                                       npy_intp *pList, float *fList,
                                       bool Wendland);

template void smDivQty<double, float>(SMX smx, npy_intp pi, int nSmooth,
                                      npy_intp *pList, float *fList,
                                      bool Wendland);

template void smMeanQty1D<float, double>(SMX smx, npy_intp pi, int nSmooth,
                                         npy_intp *pList, float *fList,
                                         bool Wendland);

template void smMeanQtyND<float, double>(SMX smx, npy_intp pi, int nSmooth,
                                         npy_intp *pList, float *fList,
                                         bool Wendland);

template void smDispQty1D<float, double>(SMX smx, npy_intp pi, int nSmooth,
                                         npy_intp *pList, float *fList,
                                         bool Wendland);

template void smDispQtyND<float, double>(SMX smx, npy_intp pi, int nSmooth,
                                         npy_intp *pList, float *fList,
                                         bool Wendland);

template void smCurlQty<float, double>(SMX smx, npy_intp pi, int nSmooth,
                                       npy_intp *pList, float *fList,
                                       bool Wendland);

template void smDivQty<float, double>(SMX smx, npy_intp pi, int nSmooth,
                                      npy_intp *pList, float *fList,
                                      bool Wendland);

template void smMeanQty1D<float, float>(SMX smx, npy_intp pi, int nSmooth,
                                        npy_intp *pList, float *fList,
                                        bool Wendland);

template void smMeanQtyND<float, float>(SMX smx, npy_intp pi, int nSmooth,
                                        npy_intp *pList, float *fList,
                                        bool Wendland);

template void smDispQty1D<float, float>(SMX smx, npy_intp pi, int nSmooth,
                                        npy_intp *pList, float *fList,
                                        bool Wendland);

template void smDispQtyND<float, float>(SMX smx, npy_intp pi, int nSmooth,
                                        npy_intp *pList, float *fList,
                                        bool Wendland);

template void smCurlQty<float, float>(SMX smx, npy_intp pi, int nSmooth,
                                      npy_intp *pList, float *fList,
                                      bool Wendland);

template void smDivQty<float, float>(SMX smx, npy_intp pi, int nSmooth,
                                     npy_intp *pList, float *fList,
                                     bool Wendland);
