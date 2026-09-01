#ifndef KD_THREADPOOL_HINCLUDED
#define KD_THREADPOOL_HINCLUDED

#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

/*
 ** A minimal fixed-size thread pool, sufficient for the kdtree build.
 **
 ** The thread that constructs the pool takes part in the work as rank 0, so a
 ** pool of n threads spawns only n-1 workers. run() hands the same callable to
 ** every rank and returns once all of them have finished; barrier() lets the
 ** ranks synchronise part-way through a run().
 **
 ** The pool is deliberately simple: there is no task queue and no work
 ** stealing. Everything the build needs is expressed as "all ranks run this
 ** function", with any load balancing done by the callable itself (see
 ** kdBuildTree, which hands out subtrees via an atomic counter).
 */
class ThreadPool {
public:
  explicit ThreadPool(int nThreads) : nThreads(nThreads < 1 ? 1 : nThreads) {
    for (int rank = 1; rank < this->nThreads; ++rank)
      workers.emplace_back(&ThreadPool::workerLoop, this, rank);
  }

  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      stopping = true;
    }
    startCv.notify_all();
    for (auto &worker : workers)
      worker.join();
  }

  ThreadPool(const ThreadPool &) = delete;
  ThreadPool &operator=(const ThreadPool &) = delete;

  int size() const { return nThreads; }

  /* Run fn(rank) on every rank, returning once they have all finished. */
  void run(std::function<void(int)> fn) {
    if (nThreads == 1) {
      fn(0);
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mutex);
      job = std::move(fn);
      nFinished = 0;
      ++generation;
    }
    startCv.notify_all();
    job(0);
    std::unique_lock<std::mutex> lock(mutex);
    doneCv.wait(lock, [this] { return nFinished == nThreads - 1; });
  }

  /* Wait for all ranks to reach this point. Only valid inside run(). */
  void barrier() {
    if (nThreads == 1)
      return;
    std::unique_lock<std::mutex> lock(barrierMutex);
    unsigned long long seen = barrierGeneration;
    if (++nWaiting == nThreads) {
      nWaiting = 0;
      ++barrierGeneration;
      barrierCv.notify_all();
    } else {
      barrierCv.wait(lock, [this, seen] { return barrierGeneration != seen; });
    }
  }

private:
  void workerLoop(int rank) {
    unsigned long long seen = 0;
    while (true) {
      std::unique_lock<std::mutex> lock(mutex);
      startCv.wait(lock, [this, seen] { return stopping || generation != seen; });
      if (stopping)
        return;
      seen = generation;
      lock.unlock();

      job(rank);

      lock.lock();
      if (++nFinished == nThreads - 1)
        doneCv.notify_one();
    }
  }

  const int nThreads;
  std::vector<std::thread> workers;
  std::function<void(int)> job;

  std::mutex mutex;
  std::condition_variable startCv, doneCv;
  unsigned long long generation = 0;
  int nFinished = 0;
  bool stopping = false;

  std::mutex barrierMutex;
  std::condition_variable barrierCv;
  unsigned long long barrierGeneration = 0;
  int nWaiting = 0;
};

/* Split [0, n) into nThreads contiguous blocks and return the rank'th. */
inline void kdRankRange(npy_intp n, int rank, int nThreads, npy_intp &begin,
                        npy_intp &end) {
  begin = (n * rank) / nThreads;
  end = (n * (rank + 1)) / nThreads;
}

#endif
