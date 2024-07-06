#pragma once

#include "kd.h"
#include "pq.h"
#include "../sph/kernels.hpp"
#include <functional>
#include <memory>
#include <mutex>
#include <stdbool.h>
#include <vector>
#include <iostream>
#include <queue>
#include <cmath>

#define RESMOOTH_SAFE 500
#define WORKUNIT 1000


#define M_1_PI 0.31830988618379067154



template<typename T>
class SmoothingContext {
public:
  KDContext* kd;

  npy_intp nSmooth;
  T fPeriod[3];

  npy_intp nListSize;
  std::vector<T> fList;
  std::vector<npy_intp> pList;

  npy_intp pin = 0, pi = 0, pNext = 0; // particle indices for distributed loops (TODO: rationalise)
  npy_intp nCurrent = 0; // particle indices for distributed loops (TODO: rationalise)

  std::shared_ptr<std::mutex> pMutex;

  SmoothingContext<T> * smx_global;


  T ax = 0.0, ay = 0.0, az = 0.0;
  bool warnings; //  keep track of whether a warning has been issued

  std::unique_ptr<std::vector<npy_intp>> result;
  std::unique_ptr<PriorityQueue<T>> priorityQueue;
  std::shared_ptr<kernels::Kernel<T>> pKernel;

  SmoothingContext(KDContext* kd, npy_intp nSmooth, T fPeriod[3]) : kd(kd), nSmooth(nSmooth), fPeriod{fPeriod[0], fPeriod[1], fPeriod[2]},
      nListSize(nSmooth + RESMOOTH_SAFE), fList(nListSize), pList(nListSize),
      pMutex(std::make_shared<std::mutex>()),
      priorityQueue(std::make_unique<PriorityQueue<T>>(nSmooth, kd->nActive)) {

  }

  SmoothingContext(const SmoothingContext<T> &copy) : kd(copy.kd), nSmooth(copy.nSmooth),
      fPeriod{copy.fPeriod[0], copy.fPeriod[1], copy.fPeriod[2]},
      nListSize(copy.nListSize), fList(nListSize), pList(nListSize), pMutex(copy.pMutex),
      smx_global(const_cast<SmoothingContext<T>*>(&copy)),
      priorityQueue(std::make_unique<PriorityQueue<T>>(nSmooth, kd->nActive)),
      pKernel(copy.pKernel) { }
      // copy constructor takes a pointer to the global context

  void setupKernel(int kernel_id) {
    pKernel = kernels::Kernel<T>::create(kernel_id, nSmooth);
  }
};


template<typename T>
bool smCheckFits(KDContext* kd, T *fPeriod) {
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


template<typename T>
void smCheckPeriodicityAndWarn(KDContext* kd, T fPeriod[3]) {
  if (!smCheckFits(kd, fPeriod)) {
    PyErr_WarnEx(PyExc_RuntimeWarning,
                 "\r\n\r\nThe particles span a region larger than the specified boxsize; disabling periodicity.\r\n\r\n"
                 "For more information about this warning, see the module documentation for KDTree, \r\n"
                 "https://pynbody.readthedocs.io/latest/reference/_autosummary/pynbody.kdtree.KDTree.html",
                 1);
    fPeriod[0] = fPeriod[1] = fPeriod[2] = std::numeric_limits<T>::max();
  }
}

template<typename T>
SmoothingContext<T> * smInit(KDContext* kd, int nSmooth, T fPeriod) {
  T fPeriodArray[3] = {fPeriod, fPeriod, fPeriod};

  if(&(kd->kdNodes[ROOT]) == nullptr) {
    PyErr_SetString(PyExc_ValueError, "Invalid KDTree");
    return nullptr;
  }

  if(nSmooth > kd->nActive) {
    PyErr_SetString(PyExc_ValueError, "nSmooth must be less than or equal to the number of particles");
    return nullptr;
  }

  smCheckPeriodicityAndWarn(kd, fPeriodArray);

  auto smx = new SmoothingContext<T>(kd, nSmooth, fPeriodArray); // not shared_ptr because python will memory manage it

  return smx;

}

template<typename T>
SmoothingContext<T> * smInitThreadLocalCopy(SmoothingContext<T> * from) {
  auto smx = new SmoothingContext<T>(*from);

  return smx;
}

template<typename T>
void smFinishThreadLocalCopy(SmoothingContext<T> * smx) {
  delete smx;
}

template<typename T>
npy_intp smGetNext(SmoothingContext<T> * smx_local) {

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


  smx_local->nCurrent += 1; // increment the local counter, ready for next time we are called

  return i;
}





template <typename T>
void smBallSearch(SmoothingContext<T> *smx, T *ri) {
  // Search for the nearest neighbors to the particle at ri[3].
  // The priority queue must already be fully populated with some candidate particles. The better candidates the
  // faster the search will perform.
  KDNode *c;
  npy_intp *p;
  KDContext* kd;

  PriorityQueue<T> *priorityQueue = smx->priorityQueue.get();

  npy_intp cell, cp, ct, pj;

  T fDist2, dx, dy, dz, lx, ly, lz, sx, sy, sz, x, y, z;

  kd = smx->kd;
  c = smx->kd->kdNodes;
  p = smx->kd->particleOffsets;

  x = ri[0];
  y = ri[1];
  z = ri[2];
  lx = smx->fPeriod[0];
  ly = smx->fPeriod[1];
  lz = smx->fPeriod[2];

  cell = kdFindLocalBucket(kd, ri);

  T fBall2 = priorityQueue->topDistanceSquaredOrMax();

  npy_intp start_particle = c[cell].pLower;
  npy_intp end_particle = c[cell].pUpper;

  for (pj = start_particle; pj <= end_particle; ++pj) {
    std::tie(dx, dy, dz) = GET2<T>(kd->pNumpyPos, p[pj]);
    dx = x-dx;
    dy = y-dy;
    dz = z-dz;

    fDist2 = dx * dx + dy * dy + dz * dz;
    priorityQueue->push(fDist2, pj);
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
        start_particle = c[cp].pLower;
        end_particle = c[cp].pUpper;

        for (pj = start_particle; pj <= end_particle; ++pj) {
          std::tie(dx, dy, dz) = GET2<T>(kd->pNumpyPos, p[pj]);
          dx = sx-dx; dy = sy-dy; dz = sz-dz;
          fDist2 = dx * dx + dy * dy + dz * dz;
          if (fDist2 < fBall2) {
            priorityQueue->push(fDist2, pj);
            fBall2 = priorityQueue->topDistanceSquaredOrMax();
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
}


template<typename T>
inline npy_intp smBallGatherStoreResultInList(SmoothingContext<T>* smx, T fDist2,
                                              npy_intp particleIndex,
                                              npy_intp foundIndex) {
  smx->result->push_back(smx->kd->particleOffsets[particleIndex]);
  return particleIndex + 1;
}

template<typename T>
inline npy_intp smBallGatherStoreResultInSmx(SmoothingContext<T>* smx, T fDist2,
                                             npy_intp particleIndex,
                                             npy_intp foundIndex) {
  if (foundIndex >= smx->nListSize) {
    if (!smx->warnings)
      fprintf(stderr, "Smooth - particle cache too small for local density - "
                      "results will be incorrect\n");
    smx->warnings = true;
    return foundIndex;
  }
  smx->fList[foundIndex] = fDist2;
  smx->pList[foundIndex] = particleIndex;
  return foundIndex + 1;
}


template <typename T,
          npy_intp (*storeResultFunction)(SmoothingContext<T> *, T, npy_intp, npy_intp)>
npy_intp smBallGather(SmoothingContext<T> * smx, T fBall2, T *ri) {
  /* Gather all particles within the specified radius, using the storeResultFunction callback
   * to store the results. */
  KDNode *c;
  npy_intp *p;
  KDContext* kd = smx->kd;
  npy_intp pj, nCnt, cp, nSplit;
  T dx, dy, dz, x, y, z, lx, ly, lz, sx, sy, sz, fDist2;

  c = smx->kd->kdNodes;
  p = smx->kd->particleOffsets;
  nSplit = smx->kd->nSplit;
  lx = smx->fPeriod[0];
  ly = smx->fPeriod[1];
  lz = smx->fPeriod[2];
  x = ri[0];
  y = ri[1];
  z = ri[2];

  // fBall2 = std::nextafter(fBall2, std::numeric_limits<T>::max());

  nCnt = 0;
  cp = ROOT;
  while (1) {
    INTERSECT(c, cp, fBall2, lx, ly, lz, x, y, z, sx, sy, sz);
    /*
     ** We have an intersection to test.
     */

    if (cp < nSplit) {
      cp = LOWER(cp);
      continue;
    } else {
      for (pj = c[cp].pLower; pj <= c[cp].pUpper; ++pj) {
        dx = sx - GET2<T>(kd->pNumpyPos, p[pj], 0);
        dy = sy - GET2<T>(kd->pNumpyPos, p[pj], 1);
        dz = sz - GET2<T>(kd->pNumpyPos, p[pj], 2);
        fDist2 = dx * dx + dy * dy + dz * dz;
        if (fDist2 <= fBall2) {
          nCnt = storeResultFunction(smx, fDist2, pj, nCnt);
        }
      }
    }
  GetNextCell:
    // called by INTERSECT when a cell can be ignored, and finds the next cell
    // to inspect
    SETNEXT(cp, ROOT);
    if (cp == ROOT)
      break;
  }
  assert(nCnt <= smx->nListSize);
  return (nCnt);
}


template <typename T>
PyObject *getReturnParticleList(SmoothingContext<T> * smx);


template <typename T>
npy_intp smSmoothStep(SmoothingContext<T> * smx, int procid) {
  KDNode *c;
  npy_intp *p;

  KDContext* kd = smx->kd;
  npy_intp pi, pin, pj, pNext, nCnt, nSmooth;
  npy_intp nScanned = 0;

  T dx, dy, dz, x, y, z, h2, ax, ay, az;
  T proc_signal = -static_cast<T>(procid)-1.0;
  T ri[3];

  c = smx->kd->kdNodes;
  p = smx->kd->particleOffsets;

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

    // Remove everything from the queue. Since the next particle may be
    // far away from the one we previously processed, a better set of candidates
    // is likely to be found by starting from the local cell, as smBallSearch will do
    // when the priority queue is empty.
    smx->priorityQueue->clear();


  } else   {
    // Calculate priority queue using existing particles
    pi = pin;

    // Mark - see comment above
    SETSMOOTH(T, pi, 10);

    x = GET2<T>(kd->pNumpyPos, p[pi], 0);
    y = GET2<T>(kd->pNumpyPos, p[pi], 1);
    z = GET2<T>(kd->pNumpyPos, p[pi], 2);

    auto priorityQueue = smx->priorityQueue.get();

    priorityQueue->updateDistances([&](PQEntry<T> &entry) {
      entry.ax -= ax;
      entry.ay -= ay;
      entry.az -= az;
      dx = x + entry.ax - GET2<T>(kd->pNumpyPos, p[entry.getParticleIndex()], 0);
      dy = y + entry.ay - GET2<T>(kd->pNumpyPos, p[entry.getParticleIndex()], 1);
      dz = z + entry.az - GET2<T>(kd->pNumpyPos, p[entry.getParticleIndex()], 2);
      entry.distanceSquared = dx * dx + dy * dy + dz * dz;
    });

    // Reset anchor coordinates
    ax = 0.0;
    ay = 0.0;
    az = 0.0;


  }

  // We are now in a situation where the priority queue has a reasonable starting list of candidates for the nearest
  // neighbours, and we can go ahead and perform a formal search
  for (int j = 0; j < 3; ++j) {
    ri[j] = GET2<T>(kd->pNumpyPos, p[pi], j);
  }

  smBallSearch<T>(smx, ri);
  SETSMOOTH(T, pi, 0.5 * sqrt(smx->priorityQueue->topDistanceSquaredOrMax()));

  // p[pi].fSmooth = 0.5*sqrt(smx->pfBall2[pi]);
  /*
  ** Pick next particle, 'pin'.
  ** Simultaneously create fList (distance info) and pList (particle indexes) for use when sending NN information
  ** back to python
  */
  pin = pi;
  nCnt = 0;
  h2 = smx->priorityQueue->topDistanceSquaredOrMax();

  smx->priorityQueue->iterateHeapEntries([&pin, &h2, &ax, &ay, &az, &nCnt, smx, kd](const PQEntry<T> &entry) {
    if (nCnt >= smx->nListSize) {
      // no room left
      if (!smx->warnings)
        fprintf(stderr, "Smooth - particle cache too small for local density - "
                        "results will be incorrect\n");
      smx->warnings = true;
      return;
    }

    smx->pList[nCnt] = entry.getParticleIndex();
    smx->fList[nCnt++] = entry.distanceSquared;

    if (GETSMOOTH(T, entry.getParticleIndex()) >= 0)
      return; // already done, don't re-do

     // Here we are setting up the next particle to be processed. For best efficiency, choose one which is as close as possible
    // to the one we just processed.
    //
    // One might imagine this would lead to attempting to proces a particle that we very recently processed already.
    // However in practice checking here that the particle is not already processed seems more expensive than just
    // processing it again. (It's not clear to me why this should be, it's just empirical.)
    if (entry.distanceSquared < h2) {
      pin = entry.getParticleIndex();
      h2 = entry.distanceSquared;
      ax = entry.ax;
      ay = entry.ay;
      az = entry.az;
    }
  });

  smx->pi = pi; // = the particle just processed, for reflection back into python when using nn_next to expose NN info
  smx->pin = pin; // = particle to be processed when smSmoothStep is next called
  smx->pNext = pNext; // = particle to start scanning from, if it turns out the next particle doesn't need processing
  smx->ax = ax; //
  smx->ay = ay; // = anchor coordinates for wrapping (?)
  smx->az = az; //

  return nCnt;
}


template <typename T>
void smSmoothInitStep(SmoothingContext<T> * smx);

template <typename T>
void smDensitySym(SmoothingContext<T> *, npy_intp, int, bool);

template <typename T>
void smDensity(SmoothingContext<T> *, npy_intp, int, bool);

template <typename Tf, typename Tq>
void smMeanQtyND(SmoothingContext<Tf> *, npy_intp, int, bool);
template <typename Tf, typename Tq>
void smDispQtyND(SmoothingContext<Tf> *, npy_intp, int, bool);
template <typename Tf, typename Tq>
void smMeanQty1D(SmoothingContext<Tf> *, npy_intp, int, bool);
template <typename Tf, typename Tq>
void smDispQty1D(SmoothingContext<Tf> *, npy_intp, int, bool);
template <typename Tf, typename Tq>
void smDivQty(SmoothingContext<Tf> *, npy_intp, int, bool);
template <typename Tf, typename Tq>
void smCurlQty(SmoothingContext<Tf> *, npy_intp, int, bool);

template <typename T> void smDomainDecomposition(KDContext* kd, int nprocs);



template<typename T>
void initParticleList(SmoothingContext<T> * smx) {
  smx->result = std::make_unique<std::vector<npy_intp>>();
  smx->result->reserve(100000);
  // not so large that it's expensive to reserve.
  // Not so small that we constantly need to get more space.
}

template<typename T>
PyObject *getReturnParticleList(SmoothingContext<T> * smx) {
  // make a numpy array from smx->result
  npy_intp dims[1] = {static_cast<npy_intp>(smx->result->size())};
  PyObject *numpy_result = PyArray_SimpleNew(1, dims, NPY_INTP);

  std::copy(smx->result->begin(), smx->result->end(),
            static_cast<long *>(
                PyArray_DATA(reinterpret_cast<PyArrayObject *>(numpy_result))));
  smx->result.reset(nullptr);

  return numpy_result;
}

template<typename T>
void smSmoothInitStep(SmoothingContext<T>* smx) {
}

template <typename T> void smDomainDecomposition(KDContext* kd, int nprocs) {

  // AP 31/8/2014 - Here is the domain decomposition for nProcs>1
  // In principle one should do better by localizing the
  // domains -- the current approach is a seriously naive decomposition.
  // This will result in more snake collisions than necessary.
  // However in practice, up to nCpu = 8, the scaling is looking
  // pretty linear anyway so I'm leaving that for future.

  // AP 6/4/2024 - revisited this while working on the improved KDTree/smoothing
  // implementation. It would seem to be more efficient to assign particles to
  // processors in 'blocks' given that the blocks are roughly spatially coherent.
  // I have made this change but in practice (as per my comment in 2014!) it seems
  // to make little difference to the performance. I also wondered whether having
  // a simpler way for the individual processors to 'scan' for their next particle
  // might be more efficient than using the smooth array in the way below. But
  // in practice so little time is spent scanning that this doesn't seem to be
  // an issue.

  npy_intp pi;

  if (nprocs > 0) {
    for (pi = 0; pi < kd->nActive; ++pi) {
      SETSMOOTH(T, pi, -static_cast<T>((pi * nprocs) / kd->nActive) - 1.0);
    }
  }
}

template<typename T>
void smInitPriorityQueue(SmoothingContext<T> * smx) {
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



template <typename T>
void smDensitySym(SmoothingContext<T> * smx, npy_intp pi, int nSmooth) {
  T fNorm, ih2, r2, rs, ih;
  npy_intp i, pj;
  KDContext* kd = smx->kd;

  ih = 1.0 / GETSMOOTH(T, pi);
  ih2 = ih * ih;
  fNorm = 0.5 * M_1_PI * ih * ih2;

  auto & kernel = *(smx->pKernel);

  for (i = 0; i < nSmooth; ++i) {
    pj = smx->pList[i];
    r2 = smx->fList[i] * ih2;
    rs = kernel(r2);
    rs *= fNorm;
    ACCUM<T>(kd->pNumpyDen, kd->particleOffsets[pi],
             rs * GET<T>(kd->pNumpyMass, kd->particleOffsets[pj]));
    ACCUM<T>(kd->pNumpyDen, kd->particleOffsets[pj],
             rs * GET<T>(kd->pNumpyMass, kd->particleOffsets[pi]));
  }
}




template <typename T>
void smDensity(SmoothingContext<T> * smx, npy_intp pi, int nSmooth) {
  T fNorm, ih2, r2, rs, ih;
  npy_intp j, pj, pi_iord;
  KDContext* kd = smx->kd;

  auto & kernel = *(smx->pKernel);

  pi_iord = kd->particleOffsets[pi];
  ih = 1.0 / GET<T>(kd->pNumpySmooth, pi_iord);
  ih2 = ih * ih;
  fNorm = M_1_PI * ih * ih2;
  SET<T>(kd->pNumpyDen, pi_iord, 0.0);
  for (j = 0; j < nSmooth; ++j) {
    pj = smx->pList[j];
    r2 = smx->fList[j] * ih2;
    rs = kernel(r2);
    rs *= fNorm;
    ACCUM<T>(kd->pNumpyDen, pi_iord,
             rs * GET<T>(kd->pNumpyMass, kd->particleOffsets[pj]));
  }
}



template <typename Tf, typename Tq>
void smMeanQty1D(SmoothingContext<Tf> * smx, npy_intp pi, int nSmooth) {
  Tf fNorm, ih2, r2, rs, ih, mass, rho;
  npy_intp j, pj, pi_iord;
  KDContext* kd = smx->kd;

  auto & kernel = *(smx->pKernel);

  pi_iord = kd->particleOffsets[pi];
  ih = 1.0 / GET<Tf>(kd->pNumpySmooth, pi_iord);
  ih2 = ih * ih;
  fNorm = M_1_PI * ih * ih2;

  SET<Tq>(kd->pNumpyQtySmoothed, pi_iord, 0.0);

  for (j = 0; j < nSmooth; ++j) {
    pj = smx->pList[j];
    r2 = smx->fList[j] * ih2;
    rs = kernel(r2);
    rs *= fNorm;
    mass = GET<Tf>(kd->pNumpyMass, kd->particleOffsets[pj]);
    rho = GET<Tf>(kd->pNumpyDen, kd->particleOffsets[pj]);
    ACCUM<Tq>(kd->pNumpyQtySmoothed, pi_iord,
              rs * mass * GET<Tq>(kd->pNumpyQty, kd->particleOffsets[pj]) / rho);
  }
}

template <typename Tf, typename Tq>
void smMeanQtyND(SmoothingContext<Tf> * smx, npy_intp pi, int nSmooth) {
  Tf fNorm, ih2, r2, rs, ih, mass, rho;
  npy_intp j, k, pj, pi_iord;
  KDContext* kd = smx->kd;

  auto & kernel = *(smx->pKernel);

  pi_iord = kd->particleOffsets[pi];
  ih = 1.0 / GET<Tf>(kd->pNumpySmooth, pi_iord);
  ih2 = ih * ih;
  fNorm = M_1_PI * ih * ih2;

  for (k = 0; k < 3; ++k)
    SET2<Tq>(kd->pNumpyQtySmoothed, pi_iord, k, 0.0);

  for (j = 0; j < nSmooth; ++j) {
    pj = smx->pList[j];
    r2 = smx->fList[j] * ih2;
    rs = kernel(r2);
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


template <typename Tf, typename Tq>
void smCurlQty(SmoothingContext<Tf> * smx, npy_intp pi, int nSmooth) {
  Tf fNorm, ih2, r2, rs, q2, ih, mass, rho, dqty[3], qty_i[3];
  npy_intp j, k, pj, pi_iord, pj_iord;
  KDContext* kd = smx->kd;
  Tf curl[3], x, y, z, dx, dy, dz;

  auto & kernel = *(smx->pKernel);

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


    // Kernel gradient
    rs = kernel.gradient(q2, r2);

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
void smDivQty(SmoothingContext<Tf> * smx, npy_intp pi, int nSmooth) {
  Tf fNorm, ih2, r2, rs, q2, ih, mass, rho, div, dqty[3], qty_i[3];
  npy_intp j, k, pj, pi_iord, pj_iord;
  KDContext* kd = smx->kd;
  Tf x, y, z, dx, dy, dz;

  auto & kernel = *(smx->pKernel);

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

    rs = kernel.gradient(q2, r2);

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
void smDispQtyND(SmoothingContext<Tf> * smx, npy_intp pi, int nSmooth) {
  Tf fNorm, ih2, r2, rs, ih, mass, rho;
  npy_intp j, k, pj, pi_iord;
  KDContext* kd = smx->kd;
  Tq mean[3], tdiff;

  auto & kernel = *(smx->pKernel);

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
    rs = kernel(r2);
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
    rs = kernel(r2);
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
void smDispQty1D(SmoothingContext<Tf> * smx, npy_intp pi, int nSmooth) {
  Tf fNorm, ih2, r2, rs, ih, mass, rho;
  npy_intp j, pj, pi_iord;
  KDContext* kd = smx->kd;
  Tq mean, tdiff;

  auto & kernel = *(smx->pKernel);

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
    rs = kernel(r2);

    rs *= fNorm;
    mass = GET<Tf>(kd->pNumpyMass, kd->particleOffsets[pj]);
    rho = GET<Tf>(kd->pNumpyDen, kd->particleOffsets[pj]);
    mean += rs * mass * GET<Tq>(kd->pNumpyQty, kd->particleOffsets[pj]) / rho;
  }

  // pass 2: get variance

  for (j = 0; j < nSmooth; ++j) {
    pj = smx->pList[j];
    r2 = smx->fList[j] * ih2;
    rs = kernel(r2);
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
