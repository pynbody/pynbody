#ifndef SMOOTH_HINCLUDED
#define SMOOTH_HINCLUDED

#include "kd.h"
#include <functional>
#include <memory>
#include <mutex>
#include <stdbool.h>
#include <vector>
#include <iostream>
#include <queue>

#define RESMOOTH_SAFE 500
#define WORKUNIT 1000


#define M_1_PI 0.31830988618379067154


template<typename T>
class PQEntry {
  protected:
    npy_intp particleIndex;

  public:
    T distanceSquared;
    T ax, ay, az;

    PQEntry(T distanceSquared, npy_intp particleIndex, T ax, T ay, T az) :
      distanceSquared(distanceSquared), particleIndex(particleIndex), ax(ax), ay(ay), az(az)  { } 

    npy_intp getParticleIndex() const { return particleIndex; }

    inline bool operator<(const PQEntry& other) const {
      return distanceSquared < other.distanceSquared;
    }

};

// output stream operator for PQEntry, for debugging:
template<typename T>
inline std::ostream& operator<<(std::ostream& os, const PQEntry<T>& pqEntry) {
  os << "PQEntry(" << pqEntry.distanceSquared << ", " << pqEntry.getParticleIndex() << ")";
  return os;
}

template<typename T>
class PriorityQueue {
  protected:
    std::vector<bool> particleIsInQueue;
    size_t maxSize;
    std::vector<PQEntry<T>> heap {};

  public:
    PriorityQueue(size_t maxSize, size_t numParticles) : maxSize(maxSize), particleIsInQueue(numParticles) {

    }

    // no copying allowed
    PriorityQueue(const PriorityQueue&) = delete;
    PriorityQueue& operator=(const PriorityQueue&) = delete;


    inline void push(T distanceSquared, npy_intp particleIndex, T ax, T ay, T az) {
      if (contains(particleIndex)) return;
      if NPY_UNLIKELY(!full()) {
        heap.push_back(PQEntry<T>(distanceSquared, particleIndex, ax, ay, az));
        std::push_heap(heap.begin(), heap.end());
        particleIsInQueue[particleIndex] = true;
      } else if (distanceSquared < topDistanceSquared()) {
        pop();

        heap.push_back(PQEntry<T>(distanceSquared, particleIndex, ax, ay, az));
        particleIsInQueue[particleIndex] = true;

        std::push_heap(heap.begin(), heap.end());
      }
    }

    bool contains(npy_intp particleIndex) const {
      return particleIsInQueue[particleIndex];
    }

    void updateDistances(std::function<void(PQEntry<T> &)> update_distance) {
      for(auto &entry : heap) {
        update_distance(entry);
      }
      std::make_heap(heap.begin(), heap.end());
    }

    void iterateHeapEntries(std::function<void(const PQEntry<T> &)> func) const {
      for(auto &entry : heap) {
        func(entry);
      }
    }

    void push(T distanceSquared, npy_intp particleIndex) {
      push(distanceSquared, particleIndex, 0.0, 0.0, 0.0);
    }

    void pop() {
      particleIsInQueue[heap.front().getParticleIndex()] = false;
      std::pop_heap(heap.begin(), heap.end());
      heap.pop_back();
    }

    const PQEntry<T>& top() const {
      return heap.front();
    }

    inline T topDistanceSquared() const {
      return heap.front().distanceSquared;
    }

    inline T topDistanceSquaredOrMax() const {
      // Return the distance squared of the top element if the queue is full, otherwise return
      // the maximum value of the type (so that all attempts to push will succeed)
      if(NPY_LIKELY(full()))
        return topDistanceSquared();
      else
        return std::numeric_limits<T>::max();
    }

    size_t size() const {
      return heap.size();
    }

    void clear() {
      iterateHeapEntries([this](const PQEntry<T> & entry) {
        this->particleIsInQueue[entry.getParticleIndex()] = 0;
      });
      heap.clear();
    }

    bool empty() const {
      return heap.empty();
    }

    inline bool full() const {
      return heap.size() == maxSize;
    }

};

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

  
  double ax = 0.0, ay = 0.0, az = 0.0;
  bool warnings; //  keep track of whether a warning has been issued

  std::unique_ptr<std::vector<npy_intp>> result;
  std::unique_ptr<PriorityQueue<T>> priorityQueue;

  SmoothingContext(KDContext* kd, npy_intp nSmooth, T fPeriod[3]) : kd(kd), nSmooth(nSmooth), fPeriod{fPeriod[0], fPeriod[1], fPeriod[2]},
      nListSize(nSmooth + RESMOOTH_SAFE), fList(nListSize), pList(nListSize), 
      pMutex(std::make_shared<std::mutex>()),
      priorityQueue(std::make_unique<PriorityQueue<T>>(nSmooth, kd->nActive)) {

  }

  SmoothingContext(const SmoothingContext<T> &copy) : kd(copy.kd), nSmooth(copy.nSmooth), 
      fPeriod{copy.fPeriod[0], copy.fPeriod[1], copy.fPeriod[2]}, 
      nListSize(copy.nListSize), fList(nListSize), pList(nListSize), pMutex(copy.pMutex), 
      smx_global(const_cast<SmoothingContext<T>*>(&copy)),
      priorityQueue(std::make_unique<PriorityQueue<T>>(nSmooth, kd->nActive)) { } 
      // copy constructor takes a pointer to the global context
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
SmoothingContext<T> * smInit(KDContext* kd, int nSmooth, T *fPeriod) {

  if(&(kd->kdNodes[ROOT]) == nullptr) {
    PyErr_SetString(PyExc_ValueError, "Invalid KDTree");
    return nullptr;
  }

  if(nSmooth > kd->nActive) {
    PyErr_SetString(PyExc_ValueError, "nSmooth must be less than or equal to the number of particles");
    return nullptr;
  }

  if(!smCheckFits<T>(kd, fPeriod)) {
    PyErr_SetString(
        PyExc_ValueError,
        "The particles span a region larger than the specified boxsize");
    return nullptr;
  }

  auto smx = new SmoothingContext<T>(kd, nSmooth, fPeriod); // not shared_ptr because python will memory manage it

  return smx;

}

using SMX = SmoothingContext<double> *;

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

template<typename T>
void smFinish(SmoothingContext<T> * smx) {
  delete smx;
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


  /*
   ** Now start the search from the bucket given by cell!
   */
  for (pj = c[cell].pLower; pj <= c[cell].pUpper; ++pj) {
    dx = x - GET2<T>(kd->pNumpyPos, p[pj], 0);
    dy = y - GET2<T>(kd->pNumpyPos, p[pj], 1);
    dz = z - GET2<T>(kd->pNumpyPos, p[pj], 2);
    fDist2 = dx * dx + dy * dy + dz * dz;
    if (fDist2 < fBall2) {
      priorityQueue->push(fDist2, pj);
      fBall2 = priorityQueue->topDistanceSquaredOrMax();
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
npy_intp smBallGather(SmoothingContext<T> * smx, float fBall2, float *ri) {
  /* Gather all particles within the specified radius, using the storeResultFunction callback
   * to store the results. */
  KDNode *c;
  npy_intp *p;
  KDContext* kd = smx->kd;
  npy_intp pj, nCnt, cp, nSplit;
  float dx, dy, dz, x, y, z, lx, ly, lz, sx, sy, sz, fDist2;

  c = smx->kd->kdNodes;
  p = smx->kd->particleOffsets;
  nSplit = smx->kd->nSplit;
  lx = smx->fPeriod[0];
  ly = smx->fPeriod[1];
  lz = smx->fPeriod[2];
  x = ri[0];
  y = ri[1];
  z = ri[2];

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

void initParticleList(SMX smx);

PyObject *getReturnParticleList(SMX smx);


template <typename T> 
npy_intp smSmoothStep(SMX smx, int procid) {
  KDNode *c;
  npy_intp *p;

  KDContext* kd = smx->kd;
  npy_intp pi, pin, pj, pNext, nCnt, nSmooth;
  npy_intp nScanned = 0;

  double dx, dy, dz, x, y, z, h2, ax, ay, az;
  float proc_signal = -(float)(procid)-1.0;
  double ri[3];

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

    priorityQueue->updateDistances([&](PQEntry<double> &entry) {
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

  smBallSearch<double>(smx, ri);
  SETSMOOTH(T, pi, 0.5 * sqrt(smx->priorityQueue->topDistanceSquared()));

  // p[pi].fSmooth = 0.5*sqrt(smx->pfBall2[pi]);
  /*
  ** Pick next particle, 'pin'.
  ** Simultaneously create fList (distance info) and pList (particle indexes) for use when sending NN information
  ** back to python
  */
  pin = pi;
  nCnt = 0;
  h2 = smx->priorityQueue->topDistanceSquared();

  smx->priorityQueue->iterateHeapEntries([&pin, &h2, &ax, &ay, &az, &nCnt, smx, kd](const PQEntry<double> &entry) {
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
    // BUT NB: mightn't this just lead to trying to go back to the one we just processed before?
    // Well, that's to sort out later...
    // 
    if (entry.distanceSquared < h2){
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


void smSmoothInitStep(SMX smx);

template <typename T>
void smDensitySym(SMX, npy_intp, int, bool);

template <typename T>
void smDensity(SMX, npy_intp, int, bool);

template <typename Tf, typename Tq>
void smMeanQtyND(SMX, npy_intp, int, bool);
template <typename Tf, typename Tq>
void smDispQtyND(SMX, npy_intp, int, bool);
template <typename Tf, typename Tq>
void smMeanQty1D(SMX, npy_intp, int, bool);
template <typename Tf, typename Tq>
void smDispQty1D(SMX, npy_intp, int, bool);
template <typename Tf, typename Tq>
void smDivQty(SMX, npy_intp, int, bool);
template <typename Tf, typename Tq>
void smCurlQty(SMX, npy_intp, int, bool);

bool smCheckFits(KDContext* kd, float *fPeriod);

template <typename T> T Wendland_kernel(SMX, T, int);

template <typename T> T cubicSpline(SMX, T);

template <typename Tf> Tf cubicSpline_gradient(Tf, Tf, Tf, Tf);

template <typename Tf> Tf Wendland_gradient(Tf, Tf);

template <typename T> void smDomainDecomposition(KDContext* kd, int nprocs);



#endif
