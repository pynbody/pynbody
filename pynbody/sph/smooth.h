#ifndef SMOOTH_HINCLUDED
#define SMOOTH_HINCLUDED

#include "kd.h"
#include <functional>
#include <memory>
#include <mutex>
#include <stdbool.h>
#include <vector>

#define RESMOOTH_SAFE 500

#define M_1_PI 0.31830988618379067154

typedef struct pqNode {
  float fKey;
  struct pqNode *pqLoser;
  struct pqNode *pqFromInt;
  struct pqNode *pqFromExt;
  struct pqNode *pqWinner; /* Only used when building initial tree */
  npy_intp p;
  float ax;
  float ay;
  float az;
} PQ;

typedef struct smContext {
  KDContext* kd;
  npy_intp nSmooth;
  float fPeriod[3];
  PQ *pq;
  PQ *pqHead;
  char *iMark;
  npy_intp nListSize;
  float *fList;
  npy_intp *pList;
  npy_intp nCurrent; // current particle index for distributed loops

  std::shared_ptr<std::mutex> pMutex;
  struct smContext *smx_global;

  npy_intp pin, pi, pNext;
  float ax, ay, az;
  bool warnings; //  keep track of whether a memory-overrun  warning has been
                 //  issued
  std::unique_ptr<std::vector<npy_intp>> result;
} *SMX;

#define PQ_INIT(pq, n)                                                         \
  {                                                                            \
    int PQ_j;                                                                  \
    if ((n) == 1) {                                                            \
      (pq)[0].pqFromInt = NULL;                                                \
      (pq)[0].pqFromExt = NULL;                                                \
    }                                                                          \
    for (PQ_j = 0; PQ_j < (n); ++PQ_j) {                                       \
      if (PQ_j < 2)                                                            \
        (pq)[PQ_j].pqFromInt = NULL;                                           \
      else                                                                     \
        (pq)[PQ_j].pqFromInt = &(pq)[PQ_j >> 1];                               \
      (pq)[PQ_j].pqFromExt = &(pq)[(PQ_j + (n)) >> 1];                         \
    }                                                                          \
  }

#define PQ_BUILD(pq, n, q)                                                     \
  {                                                                            \
    int PQ_i, PQ_j;                                                            \
    PQ *PQ_t, *PQ_lt;                                                          \
    for (PQ_j = (n)-1; PQ_j > 0; --PQ_j) {                                     \
      PQ_i = (PQ_j << 1);                                                      \
      if (PQ_i < (n))                                                          \
        PQ_t = (pq)[PQ_i].pqWinner;                                            \
      else                                                                     \
        PQ_t = &(pq)[PQ_i - (n)];                                              \
      ++PQ_i;                                                                  \
      if (PQ_i < (n))                                                          \
        PQ_lt = (pq)[PQ_i].pqWinner;                                           \
      else                                                                     \
        PQ_lt = &(pq)[PQ_i - (n)];                                             \
      if (PQ_t->fKey < PQ_lt->fKey) {                                          \
        (pq)[PQ_j].pqLoser = PQ_t;                                             \
        (pq)[PQ_j].pqWinner = PQ_lt;                                           \
      } else {                                                                 \
        (pq)[PQ_j].pqLoser = PQ_lt;                                            \
        (pq)[PQ_j].pqWinner = PQ_t;                                            \
      }                                                                        \
    }                                                                          \
    if ((n) == 1)                                                              \
      (q) = (pq);                                                              \
    else                                                                       \
      (q) = (pq)[1].pqWinner;                                                  \
  }

#define PQ_REPLACE(q)                                                          \
  {                                                                            \
    PQ *PQ_t, *PQ_lt;                                                          \
    PQ_t = (q)->pqFromExt;                                                     \
    while (PQ_t) {                                                             \
      if (PQ_t->pqLoser->fKey > (q)->fKey) {                                   \
        PQ_lt = PQ_t->pqLoser;                                                 \
        PQ_t->pqLoser = (q);                                                   \
        (q) = PQ_lt;                                                           \
      }                                                                        \
      PQ_t = PQ_t->pqFromInt;                                                  \
    }                                                                          \
  }

double M3(double);
double dM3(double);
double F3(double);
double dF3(double);
double K3(double);
double dK3(double);

int smInit(SMX *, KDContext *, int, float *);
void smInitPriorityQueue(SMX);
void smFinish(SMX);

template <typename T> void smBallSearch(SMX, float, float *);

inline npy_intp smBallGatherStoreResultInList(SMX smx, float fDist2,
                                              npy_intp particleIndex,
                                              npy_intp foundIndex) {
  smx->result->push_back(smx->kd->particleOffsets[particleIndex]);
  return particleIndex + 1;
}

inline npy_intp smBallGatherStoreResultInSmx(SMX smx, float fDist2,
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
          npy_intp (*storeResultFunction)(SMX, float, npy_intp, npy_intp)>
npy_intp smBallGather(SMX smx, float fBall2, float *ri) {
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

template <typename T> npy_intp smSmoothStep(SMX smx, int procid);

void smSmoothInitStep(SMX smx, int nProcs);

template <typename T>
void smDensitySym(SMX, npy_intp, int, npy_intp *, float *, bool);

template <typename T>
void smDensity(SMX, npy_intp, int, npy_intp *, float *, bool);

template <typename Tf, typename Tq>
void smMeanQtyND(SMX, npy_intp, int, npy_intp *, float *, bool);
template <typename Tf, typename Tq>
void smDispQtyND(SMX, npy_intp, int, npy_intp *, float *, bool);
template <typename Tf, typename Tq>
void smMeanQty1D(SMX, npy_intp, int, npy_intp *, float *, bool);
template <typename Tf, typename Tq>
void smDispQty1D(SMX, npy_intp, int, npy_intp *, float *, bool);
template <typename Tf, typename Tq>
void smDivQty(SMX, npy_intp, int, npy_intp *, float *, bool);
template <typename Tf, typename Tq>
void smCurlQty(SMX, npy_intp, int, npy_intp *, float *, bool);

bool smCheckFits(KDContext* kd, float *fPeriod);

template <typename T> T Wendland_kernel(SMX, T, int);

template <typename T> T cubicSpline(SMX, T);

template <typename Tf> Tf cubicSpline_gradient(Tf, Tf, Tf, Tf);

template <typename Tf> Tf Wendland_gradient(Tf, Tf);

template <typename T> void smDomainDecomposition(KDContext* kd, int nprocs);

npy_intp smGetNext(SMX smx_local);

SMX smInitThreadLocalCopy(SMX smx_global);
void smFinishThreadLocalCopy(SMX smx_local);

#endif
