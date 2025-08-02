#ifndef KD_HINCLUDED
#define KD_HINCLUDED

#include <tuple>

#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL PYNBODY_ARRAY_API
#include <numpy/ndarraytypes.h>
#include <numpy/ndarrayobject.h>

#define ROOT 1
#define LOWER(i) (i << 1)
#define UPPER(i) ((i << 1) + 1)
#define PARENT(i) (i >> 1)
#define SIBLING(i) ((i & 1) ? i - 1 : i + 1)
#define SETNEXT(i, localroot)                                                  \
  {                                                                            \
    while (i & 1 && i != localroot)                                            \
      i = i >> 1;                                                              \
    if (i != localroot)                                                        \
      ++i;                                                                     \
  }

#define DARK 1
#define GAS 2
#define STAR 4

struct Boundary {
  double fMin[3];
  double fMax[3];
};

struct KDNode {
  double fSplit;
  Boundary bnd;
  int iDim;
  npy_intp pLower;
  npy_intp pUpper;
};

struct KDContext {
  npy_intp nBucket;
  npy_intp nParticles;
  npy_intp nActive;
  int nLevels;
  npy_intp nNodes;
  npy_intp nSplit;

  npy_intp *particleOffsets; // length N array mapping from KDTree order to pynbody/file order
  PyArrayObject *pNumpyParticleOffsets; // Numpy array from which the pointer stored in particleOffsets has been derived

  KDNode *kdNodes;           // length-N_nodes array of KDNode
  PyArrayObject *kdNodesPyArrayObject; // Numpy array from which the pointer stored in kdNodes has been derived


  int nBitDepth;        // bit depth of arrays other than pNumpyQty which can be
                        // different
  PyArrayObject *pNumpyPos;  // Nx3 Numpy array of positions
  PyArrayObject *pNumpyMass; // Nx1 Numpy array of masses
  PyArrayObject *pNumpySmooth;
  PyArrayObject *pNumpyDen;         // Nx1 Numpy array of density
  PyArrayObject *pNumpyQty;         // Nx1 Numpy array of density
  PyArrayObject *pNumpyQtySmoothed; // Nx1 Numpy array of density

};

#define INTERSECT(c, cp, fBall2, lx, ly, lz, x, y, z, sx, sy, sz)              \
  {                                                                            \
    double INTRSCT_dx, INTRSCT_dy, INTRSCT_dz;                                 \
    double INTRSCT_dx1, INTRSCT_dy1, INTRSCT_dz1, INTRSCT_fDist2;              \
    INTRSCT_dx = c[cp].bnd.fMin[0] - x;                                        \
    INTRSCT_dx1 = x - c[cp].bnd.fMax[0];                                       \
    INTRSCT_dy = c[cp].bnd.fMin[1] - y;                                        \
    INTRSCT_dy1 = y - c[cp].bnd.fMax[1];                                       \
    INTRSCT_dz = c[cp].bnd.fMin[2] - z;                                        \
    INTRSCT_dz1 = z - c[cp].bnd.fMax[2];                                       \
    if (INTRSCT_dx > 0.0) {                                                    \
      INTRSCT_dx1 += lx;                                                       \
      if (INTRSCT_dx1 < INTRSCT_dx) {                                          \
        INTRSCT_fDist2 = INTRSCT_dx1 * INTRSCT_dx1;                            \
        sx = x + lx;                                                           \
      } else {                                                                 \
        INTRSCT_fDist2 = INTRSCT_dx * INTRSCT_dx;                              \
        sx = x;                                                                \
      }                                                                        \
      if (INTRSCT_fDist2 > fBall2)                                             \
        goto GetNextCell;                                                      \
    } else if (INTRSCT_dx1 > 0.0) {                                            \
      INTRSCT_dx += lx;                                                        \
      if (INTRSCT_dx < INTRSCT_dx1) {                                          \
        INTRSCT_fDist2 = INTRSCT_dx * INTRSCT_dx;                              \
        sx = x - lx;                                                           \
      } else {                                                                 \
        INTRSCT_fDist2 = INTRSCT_dx1 * INTRSCT_dx1;                            \
        sx = x;                                                                \
      }                                                                        \
      if (INTRSCT_fDist2 > fBall2)                                             \
        goto GetNextCell;                                                      \
    } else {                                                                   \
      INTRSCT_fDist2 = 0.0;                                                    \
      sx = x;                                                                  \
    }                                                                          \
    if (INTRSCT_dy > 0.0) {                                                    \
      INTRSCT_dy1 += ly;                                                       \
      if (INTRSCT_dy1 < INTRSCT_dy) {                                          \
        INTRSCT_fDist2 += INTRSCT_dy1 * INTRSCT_dy1;                           \
        sy = y + ly;                                                           \
      } else {                                                                 \
        INTRSCT_fDist2 += INTRSCT_dy * INTRSCT_dy;                             \
        sy = y;                                                                \
      }                                                                        \
      if (INTRSCT_fDist2 > fBall2)                                             \
        goto GetNextCell;                                                      \
    } else if (INTRSCT_dy1 > 0.0) {                                            \
      INTRSCT_dy += ly;                                                        \
      if (INTRSCT_dy < INTRSCT_dy1) {                                          \
        INTRSCT_fDist2 += INTRSCT_dy * INTRSCT_dy;                             \
        sy = y - ly;                                                           \
      } else {                                                                 \
        INTRSCT_fDist2 += INTRSCT_dy1 * INTRSCT_dy1;                           \
        sy = y;                                                                \
      }                                                                        \
      if (INTRSCT_fDist2 > fBall2)                                             \
        goto GetNextCell;                                                      \
    } else {                                                                   \
      sy = y;                                                                  \
    }                                                                          \
    if (INTRSCT_dz > 0.0) {                                                    \
      INTRSCT_dz1 += lz;                                                       \
      if (INTRSCT_dz1 < INTRSCT_dz) {                                          \
        INTRSCT_fDist2 += INTRSCT_dz1 * INTRSCT_dz1;                           \
        sz = z + lz;                                                           \
      } else {                                                                 \
        INTRSCT_fDist2 += INTRSCT_dz * INTRSCT_dz;                             \
        sz = z;                                                                \
      }                                                                        \
      if (INTRSCT_fDist2 > fBall2)                                             \
        goto GetNextCell;                                                      \
    } else if (INTRSCT_dz1 > 0.0) {                                            \
      INTRSCT_dz += lz;                                                        \
      if (INTRSCT_dz < INTRSCT_dz1) {                                          \
        INTRSCT_fDist2 += INTRSCT_dz * INTRSCT_dz;                             \
        sz = z - lz;                                                           \
      } else {                                                                 \
        INTRSCT_fDist2 += INTRSCT_dz1 * INTRSCT_dz1;                           \
        sz = z;                                                                \
      }                                                                        \
      if (INTRSCT_fDist2 > fBall2)                                             \
        goto GetNextCell;                                                      \
    } else {                                                                   \
      sz = z;                                                                  \
    }                                                                          \
  }

void kdCountNodes(KDContext *kd);

template <typename T> void kdBuildTree(KDContext*, int num_threads);
template <typename T> void kdBuildNode(KDContext*, npy_intp, int);

void kdCombine(KDNode *p1, KDNode *p2, KDNode *pOut);

template <typename T> T GET(PyArrayObject *ar, npy_intp i) {
  return *((T *)PyArray_GETPTR1(ar, i));
}

template <typename T> T GET2(PyArrayObject *ar, npy_intp i, npy_intp j) {
  return *((T *)PyArray_GETPTR2(ar, i, j));
}

template <typename T> std::tuple<T, T, T> GET2(PyArrayObject *ar, npy_intp i) {
  T* ptr = (T *)PyArray_GETPTR1(ar, i);
  return std::make_tuple(ptr[0], ptr[1], ptr[2]);
}

template <typename T> void SET(PyArrayObject *ar, npy_intp i, T val) {
  *((T *)PyArray_GETPTR1(ar, i)) = val;
}

template <typename T> void SET2(PyArrayObject *ar, npy_intp i, npy_intp j, T val) {
  *((T *)PyArray_GETPTR2(ar, i, j)) = val;
}

template <typename T> void ACCUM(PyArrayObject *ar, npy_intp i, T val) {
  (*((T *)PyArray_GETPTR1(ar, i))) += val;
}

template <typename T> void ACCUM2(PyArrayObject *ar, npy_intp i, npy_intp j, T val) {
  (*((T *)PyArray_GETPTR2(ar, i, j))) += val;
}

template <typename T>
inline npy_intp kdFindLocalBucket(KDContext *kdtree, T *position) {
  npy_intp cell = ROOT;
  KDNode *c = kdtree->kdNodes;
  while (cell < kdtree->nSplit) {
    if (position[c[cell].iDim] < c[cell].fSplit)
      cell = LOWER(cell);
    else
      cell = UPPER(cell);
  }
  return cell;
}

#define GETSMOOTH(T, pid) GET<T>(kd->pNumpySmooth, kd->particleOffsets[pid])
#define SETSMOOTH(T, pid, val) SET<T>(kd->pNumpySmooth, kd->particleOffsets[pid], val)

#endif
