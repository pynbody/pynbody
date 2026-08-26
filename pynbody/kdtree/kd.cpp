#include <algorithm>
#include <assert.h>
#include <atomic>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <memory>
#include <stdlib.h>
#include <vector>

#define NO_IMPORT_ARRAY
#include "kd.h"
#include "threadpool.h"

void kdCombine(KDNode *p1, KDNode *p2, KDNode *pOut) {
  int j;

  /*
   ** Combine the bounds.
   */
  for (j = 0; j < 3; ++j) {
    if (p2->bnd.fMin[j] < p1->bnd.fMin[j])
      pOut->bnd.fMin[j] = p2->bnd.fMin[j];
    else
      pOut->bnd.fMin[j] = p1->bnd.fMin[j];
    if (p2->bnd.fMax[j] > p1->bnd.fMax[j])
      pOut->bnd.fMax[j] = p2->bnd.fMax[j];
    else
      pOut->bnd.fMax[j] = p1->bnd.fMax[j];
  }
}

/*
 ** Partition p[l..r] about the k'th smallest particle in dimension d, so that
 ** everything at or below k lies on the low side of it and everything above k
 ** on the high side.
 */
template <typename T>
void kdSelect(KDContext* kd, npy_intp d, npy_intp k, npy_intp l, npy_intp r) {
  npy_intp *p, t;
  T v;
  npy_intp i, j;

  p = kd->particleOffsets;
  while (r > l) {
    v = GET2<T>(kd->pNumpyPos, p[k], d);
    t = p[r];
    p[r] = p[k];
    p[k] = t;
    i = l - 1;
    j = r;
    while (1) {
      while (i < j)
        if (GET2<T>(kd->pNumpyPos, p[++i], d) >= v)
          break;
      while (i < j)
        if (GET2<T>(kd->pNumpyPos, p[--j], d) <= v)
          break;
      t = p[i];
      p[i] = p[j];
      p[j] = t;
      if (j <= i)
        break;
    }
    p[j] = p[i];
    p[i] = p[r];
    p[r] = t;
    if (i >= k)
      r = i - 1;
    if (i <= k)
      l = i + 1;
  }
}

template <typename T> void kdUpPass(KDContext* kd, npy_intp iCell) {
  KDNode *c;
  npy_intp l, u, pj, j;
  double rj;
  c = kd->kdNodes;
  if (c[iCell].iDim != -1) {
    l = LOWER(iCell);
    u = UPPER(iCell);
    kdUpPass<T>(kd, l);
    kdUpPass<T>(kd, u);
    kdCombine(&c[l], &c[u], &c[iCell]);
  } else {
    l = c[iCell].pLower;
    u = c[iCell].pUpper;
    // Which particles land in this leaf does not depend on how the build was
    // shared out between threads, but the order the partitioning leaves them
    // in does. Sorting settles it, so that particleOffsets comes out the same
    // whatever num_threads was, and has the loop below walk the position
    // array in ascending order.
    std::sort(kd->particleOffsets + l, kd->particleOffsets + u + 1);
    for (j = 0; j < 3; ++j) {
      c[iCell].bnd.fMin[j] = GET2<T>(kd->pNumpyPos, kd->particleOffsets[u], j);
      c[iCell].bnd.fMax[j] = c[iCell].bnd.fMin[j];
    }
    for (pj = l; pj < u; ++pj) {
      for (j = 0; j < 3; ++j) {
        rj = GET2<T>(kd->pNumpyPos, kd->particleOffsets[pj], j);
        if (rj < c[iCell].bnd.fMin[j])
          c[iCell].bnd.fMin[j] = rj;
        if (rj > c[iCell].bnd.fMax[j])
          c[iCell].bnd.fMax[j] = rj;
      }
    }
  }
}

void kdCountNodes(KDContext *kd) {
  npy_intp l, n;

  n = kd->nActive;
  kd->nLevels = 1;
  l = 1;
  while (n > kd->nBucket) {
    n = n >> 1;
    l = l << 1;
    ++kd->nLevels;
  }
  kd->nSplit = l;
  kd->nNodes = l << 1;
}

/*
 ** Work out how node iCell should be split, or return false if it is a leaf.
 ** On success d is the splitting dimension and m the index of the median
 ** particle within the node's range.
 */
static bool kdPrepareSplit(KDContext *kd, npy_intp iCell, npy_intp &d, npy_intp &m) {
  KDNode *nodes = kd->kdNodes;
  assert(nodes[iCell].pUpper - nodes[iCell].pLower + 1 > 0);
  if (!(iCell < kd->nSplit && (nodes[iCell].pUpper - nodes[iCell].pLower) > 0)) {
    nodes[iCell].iDim = -1;
    return false;
  }

  // Select splitting dimension on the basis of keeping things as square as
  // possible
  d = 0;
  for (npy_intp j = 1; j < 3; ++j) {
    if (nodes[iCell].bnd.fMax[j] - nodes[iCell].bnd.fMin[j] >
        nodes[iCell].bnd.fMax[d] - nodes[iCell].bnd.fMin[d])
      d = j;
  }
  nodes[iCell].iDim = d;

  // Find mid-point of particle list at which splitting will ultimately take
  // place
  m = (nodes[iCell].pLower + nodes[iCell].pUpper) / 2;
  return true;
}

/*
 ** Record the split of node iCell and set up its two children. The particle
 ** list must already have been partitioned about m (see kdSelect).
 */
template <typename T>
static void kdRecordSplit(KDContext *kd, npy_intp iCell, npy_intp d, npy_intp m) {
  KDNode *nodes = kd->kdNodes;

  // Note split point based on median particle
  nodes[iCell].fSplit = GET2<T>(kd->pNumpyPos, kd->particleOffsets[m], d);

  // Set up lower cell
  nodes[LOWER(iCell)].bnd = nodes[iCell].bnd;
  nodes[LOWER(iCell)].bnd.fMax[d] = nodes[iCell].fSplit;
  nodes[LOWER(iCell)].pLower = nodes[iCell].pLower;
  nodes[LOWER(iCell)].pUpper = m;

  // Set up upper cell
  nodes[UPPER(iCell)].bnd = nodes[iCell].bnd;
  nodes[UPPER(iCell)].bnd.fMin[d] = nodes[iCell].fSplit;
  nodes[UPPER(iCell)].pLower = m + 1;
  nodes[UPPER(iCell)].pUpper = nodes[iCell].pUpper;

  npy_intp diff = (m - nodes[iCell].pLower + 1) - (nodes[iCell].pUpper - m);
  assert(diff == 0 || diff == 1);
  (void)diff;
}

/* Bounding box of particles at offsets [begin, end). */
template <typename T>
static Boundary kdBounds(KDContext *kd, npy_intp begin, npy_intp end) {
  Boundary bnd;
  for (npy_intp j = 0; j < 3; ++j) {
    bnd.fMin[j] = HUGE_VAL;
    bnd.fMax[j] = -HUGE_VAL;
  }
  for (npy_intp i = begin; i < end; ++i) {
    for (npy_intp j = 0; j < 3; ++j) {
      T rj = GET2<T>(kd->pNumpyPos, kd->particleOffsets[i], j);
      if (rj < bnd.fMin[j])
        bnd.fMin[j] = rj;
      if (rj > bnd.fMax[j])
        bnd.fMax[j] = rj;
    }
  }
  return bnd;
}

static void kdExpandBounds(Boundary &bnd, const Boundary &other) {
  for (npy_intp j = 0; j < 3; ++j) {
    if (other.fMin[j] < bnd.fMin[j])
      bnd.fMin[j] = other.fMin[j];
    if (other.fMax[j] > bnd.fMax[j])
      bnd.fMax[j] = other.fMax[j];
  }
}

/*
 ** Scratch space for a selection that all threads work on together. The buffer
 ** holds the partitioned copy of the range being worked on, and is therefore
 ** as long as the particle list; it is only allocated while the top of the
 ** tree is being built (see kdBuildTree).
 */
namespace {
struct SelectWorkspace {
  std::unique_ptr<npy_intp[]> buffer; // deliberately left uninitialised
  std::vector<npy_intp> nBelow, nEqual, nAbove; // per-rank counts
};
} // namespace

static inline uint64_t kdRandom(uint64_t &state) {
  // splitmix64; we only need something cheap that doesn't resonate with
  // regular particle grids
  state += 0x9e3779b97f4a7c15ull;
  uint64_t z = state;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}

/*
 ** As kdSelect, but with every thread in the pool working on the same range.
 **
 ** The range is repeatedly partitioned three ways about a sampled pivot ---
 ** below it, equal to it, above it --- with each thread responsible for a
 ** contiguous block: every thread counts its own block, the counts are
 ** combined to tell each thread where to write, and each thread then copies
 ** its block into the shared buffer. Only the part of the range that still
 ** contains the median is carried into the next iteration, so this converges
 ** in a handful of passes, and the range is down to the serial kdSelect long
 ** before the synchronisation cost becomes comparable to the work done
 ** between barriers.
 **
 ** The pivot is one of the particles in the range, so the equal block is never
 ** empty and the range therefore strictly shrinks each time round. If the
 ** median falls inside the equal block there is nothing left to decide: every
 ** particle there shares the split coordinate.
 */
template <typename T>
static void kdSelectCooperative(KDContext *kd, ThreadPool &pool,
                                SelectWorkspace &ws, npy_intp d, npy_intp k,
                                npy_intp l, npy_intp r) {
  const npy_intp MIN_COOPERATIVE_RANGE = 1 << 14;
  const int nThreads = pool.size();
  npy_intp *p = kd->particleOffsets;
  npy_intp *buffer = ws.buffer.get();
  uint64_t rng = 0x5eed5eed5eed5eedull ^ (uint64_t)l ^ ((uint64_t)r << 20) ^
                 ((uint64_t)d << 40);

  while (r - l > MIN_COOPERATIVE_RANGE) {
    // Pivot from a small random sample, aimed at the rank we actually want.
    // Sampling at random rather than at a fixed stride matters for particles
    // laid out on a grid, where a stride can hit the same coordinate every time.
    const int nSample = 63;
    T sample[nSample];
    for (int i = 0; i < nSample; ++i)
      sample[i] = GET2<T>(kd->pNumpyPos,
                          p[l + (npy_intp)(kdRandom(rng) % (uint64_t)(r - l + 1))], d);
    std::sort(sample, sample + nSample);
    npy_intp iSample = (npy_intp)((double(k - l) / double(r - l)) * nSample);
    if (iSample < 0)
      iSample = 0;
    if (iSample >= nSample)
      iSample = nSample - 1;
    const T v = sample[iSample];

    pool.run([&](int rank) {
      npy_intp begin, end;
      kdRankRange(r - l + 1, rank, nThreads, begin, end);
      begin += l;
      end += l;

      npy_intp nBelow = 0, nEqual = 0, nAbove = 0;
      for (npy_intp i = begin; i < end; ++i) {
        T x = GET2<T>(kd->pNumpyPos, p[i], d);
        if (x < v)
          ++nBelow;
        else if (x > v)
          ++nAbove;
        else
          ++nEqual;
      }
      ws.nBelow[rank] = nBelow;
      ws.nEqual[rank] = nEqual;
      ws.nAbove[rank] = nAbove;

      pool.barrier();

      npy_intp totalBelow = 0, totalEqual = 0;
      for (int i = 0; i < nThreads; ++i) {
        totalBelow += ws.nBelow[i];
        totalEqual += ws.nEqual[i];
      }
      npy_intp offsetBelow = 0, offsetEqual = 0, offsetAbove = 0;
      for (int i = 0; i < rank; ++i) {
        offsetBelow += ws.nBelow[i];
        offsetEqual += ws.nEqual[i];
        offsetAbove += ws.nAbove[i];
      }
      npy_intp writeBelow = l + offsetBelow;
      npy_intp writeEqual = l + totalBelow + offsetEqual;
      npy_intp writeAbove = l + totalBelow + totalEqual + offsetAbove;
      for (npy_intp i = begin; i < end; ++i) {
        T x = GET2<T>(kd->pNumpyPos, p[i], d);
        if (x < v)
          buffer[writeBelow++] = p[i];
        else if (x > v)
          buffer[writeAbove++] = p[i];
        else
          buffer[writeEqual++] = p[i];
      }

      pool.barrier();

      for (npy_intp i = begin; i < end; ++i)
        p[i] = buffer[i];
    });

    npy_intp totalBelow = 0, totalEqual = 0;
    for (int i = 0; i < nThreads; ++i) {
      totalBelow += ws.nBelow[i];
      totalEqual += ws.nEqual[i];
    }
    if (k < l + totalBelow) {
      r = l + totalBelow - 1;
    } else if (k < l + totalBelow + totalEqual) {
      // k lies among the particles sharing the pivot's coordinate, so the
      // range is already partitioned about the median
      return;
    } else {
      l = l + totalBelow + totalEqual;
    }
  }

  kdSelect<T>(kd, d, k, l, r);
}

/*
 ** Build the tree, using num_threads threads.
 **
 ** The top of the tree has too few nodes to give every thread one each, so
 ** there the threads cooperate on each node's median selection in turn
 ** (kdSelectCooperative). Once there are enough nodes to go round, each node
 ** is handled by a single thread, and below that the independent subtrees are
 ** handed out through an atomic counter so that any number of threads --- not
 ** just a power of two --- is kept busy to the end of the build.
 */
template <typename T> void kdBuildTree(KDContext* kd, int num_threads) {
  // start by assuming kdCountNodes(kd) has been called and kd->kdNodes!=NULL
  assert(kd->nNodes > 0);
  assert(kd->kdNodes != NULL);

  // A thread is only ever any use if there is a subtree to give it, so there
  // is nothing to be gained from a pool bigger than the number of leaves. In
  // particular a tree that is a single leaf spawns no threads at all.
  if (num_threads > kd->nSplit)
    num_threads = (int)kd->nSplit;

  ThreadPool pool(num_threads);
  const int nThreads = pool.size();

  // Set up the initial ordering, and the bounding box of all particles
  Boundary bnd;
  if (nThreads == 1) {
    for (npy_intp i = 0; i < kd->nActive; ++i)
      kd->particleOffsets[i] = i;
    bnd = kdBounds<T>(kd, 0, kd->nActive);
  } else {
    std::vector<Boundary> partial(nThreads);
    pool.run([&](int rank) {
      npy_intp begin, end;
      kdRankRange(kd->nActive, rank, nThreads, begin, end);
      for (npy_intp i = begin; i < end; ++i)
        kd->particleOffsets[i] = i;
      partial[rank] = kdBounds<T>(kd, begin, end);
    });
    bnd = partial[0];
    for (int rank = 1; rank < nThreads; ++rank)
      kdExpandBounds(bnd, partial[rank]);
  }

  if (kd->nActive == 0) {
    kd->kdNodes[ROOT].pLower = 0;
    kd->kdNodes[ROOT].pUpper = -1;
    kd->kdNodes[ROOT].iDim = -1;
    return;
  }

  // Set up root node
  kd->kdNodes[ROOT].pLower = 0;
  kd->kdNodes[ROOT].pUpper = kd->nActive - 1;
  kd->kdNodes[ROOT].bnd = bnd;

  if (nThreads == 1) {
    kdBuildNode<T>(kd, ROOT);
    kdUpPass<T>(kd, ROOT);
    return;
  }

  // Depth at which nodes become independent tasks. Several times more tasks
  // than threads keeps every thread busy whatever the thread count, at
  // negligible cost since the levels above are parallel anyway.
  int nTreeLevels = 0;
  for (npy_intp n = kd->nSplit; n > 1; n >>= 1)
    ++nTreeLevels;
  int taskLevel = 0;
  while ((npy_intp(1) << taskLevel) < 8 * (npy_intp)nThreads && taskLevel < nTreeLevels)
    ++taskLevel;

  // A node is only built if all of its ancestors were split; the rest of the
  // node array is left untouched, exactly as in a serial build.
  std::vector<char> reached((size_t)2 << taskLevel, 0);
  reached[ROOT] = 1;

  {
    // The cooperative selection needs somewhere to build the partitioned copy
    // of a range, which for the root node means the whole particle list: one
    // extra npy_intp per particle, held only for as long as the top of the
    // tree is being built. If it cannot be had, fall back to splitting the top
    // nodes one thread at a time, which is slower but still correct.
    SelectWorkspace ws;
    ws.buffer.reset(new (std::nothrow) npy_intp[kd->nActive]);
    ws.nBelow.resize(nThreads);
    ws.nEqual.resize(nThreads);
    ws.nAbove.resize(nThreads);

    for (int level = 0; level < taskLevel; ++level) {
      const npy_intp first = npy_intp(1) << level;

      if (first < nThreads && ws.buffer) {
        // Too few nodes to go round: all threads work on each in turn
        for (npy_intp i = first; i < 2 * first; ++i) {
          npy_intp d, m;
          if (!reached[i] || !kdPrepareSplit(kd, i, d, m))
            continue;
          kdSelectCooperative<T>(kd, pool, ws, d, m, kd->kdNodes[i].pLower,
                                 kd->kdNodes[i].pUpper);
          kdRecordSplit<T>(kd, i, d, m);
          reached[LOWER(i)] = reached[UPPER(i)] = 1;
        }
      } else {
        // One node per thread; contiguous blocks so that a thread stays on its
        // own part of the particle list from one level to the next
        pool.run([&](int rank) {
          npy_intp begin, end;
          kdRankRange(first, rank, nThreads, begin, end);
          for (npy_intp i = first + begin; i < first + end; ++i) {
            npy_intp d, m;
            if (!reached[i] || !kdPrepareSplit(kd, i, d, m))
              continue;
            kdSelect<T>(kd, d, m, kd->kdNodes[i].pLower, kd->kdNodes[i].pUpper);
            kdRecordSplit<T>(kd, i, d, m);
            reached[LOWER(i)] = reached[UPPER(i)] = 1;
          }
        });
      }
    }
  } // the selection buffer is no longer needed

  // Each remaining subtree is independent, so hand them out one at a time
  std::vector<npy_intp> tasks;
  for (npy_intp i = npy_intp(1) << taskLevel; i < npy_intp(2) << taskLevel; ++i)
    if (reached[i])
      tasks.push_back(i);

  std::atomic<size_t> nextTask(0);
  pool.run([&](int rank) {
    while (true) {
      size_t task = nextTask.fetch_add(1);
      if (task >= tasks.size())
        break;
      kdBuildNode<T>(kd, tasks[task]);
      kdUpPass<T>(kd, tasks[task]);
    }
  });

  // The subtrees have passed their bounds up as far as taskLevel; fold the
  // remaining few levels together here.
  for (int level = taskLevel - 1; level >= 0; --level) {
    for (npy_intp i = npy_intp(1) << level; i < npy_intp(2) << level; ++i) {
      if (!reached[i])
        continue;
      if (kd->kdNodes[i].iDim != -1)
        kdCombine(&kd->kdNodes[LOWER(i)], &kd->kdNodes[UPPER(i)], &kd->kdNodes[i]);
      else
        kdUpPass<T>(kd, i); // a leaf this high up: work its bounds out directly
    }
  }
}

template <typename T>
void kdBuildNode(KDContext* kd, npy_intp local_root) {

  npy_intp i = local_root;
  npy_intp d, m;
  KDNode *nodes;
  nodes = kd->kdNodes;

  while (1) {
    if (kdPrepareSplit(kd, i, d, m)) {

      // Sort list to ensure particles between lower and m are to
      // the 'left' of particles between m and upper
      kdSelect<T>(kd, d, m, nodes[i].pLower, nodes[i].pUpper);

      kdRecordSplit<T>(kd, i, d, m);

      // Next cell is the lower one. Upper one will be processed
      // on the way up.
      i = LOWER(i);

    } else {
      // Cell does not need to be split; kdPrepareSplit has marked it as a
      // leaf. Go back up the tree and process the UPPER cells where necessary
      SETNEXT(i, local_root);
    }
    if (i == local_root)
      break; // We got back to the top, so we're done.
  }
}

// instantiate the actual functions that are available:

template void kdSelect<double>(KDContext* kd, npy_intp d, npy_intp k, npy_intp l,
                               npy_intp r);

template void kdUpPass<double>(KDContext* kd, npy_intp iCell);

template void kdBuildTree<double>(KDContext* kd, int num_threads);

template void kdBuildNode<double>(KDContext* kd, npy_intp local_root);

template void kdSelect<float>(KDContext* kd, npy_intp d, npy_intp k, npy_intp l,
                              npy_intp r);

template void kdUpPass<float>(KDContext* kd, npy_intp iCell);

template void kdBuildTree<float>(KDContext* kd, int num_threads);

template void kdBuildNode<float>(KDContext* kd, npy_intp local_root);
