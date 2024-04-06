#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>
#include <limits>
#include <memory>

template <typename DifferenceT>
DifferenceT heap_parent(DifferenceT k)
{
    return (k - 1) / 2;
}

template <typename DifferenceT>
DifferenceT heap_left(DifferenceT k)
{
    return 2 * k + 1;
}


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

template<typename T>
struct PQEntryPtrComparator {
  bool operator()(const std::unique_ptr<PQEntry<T>> & lhs, const std::unique_ptr<PQEntry<T>> & rhs) {
    return lhs->distanceSquared < rhs->distanceSquared;
  }
};

// output stream operator for PQEntry, for debugging:
template<typename T>
inline std::ostream& operator<<(std::ostream& os, const PQEntry<T>& pqEntry) {
  os << "PQEntry(" << pqEntry.distanceSquared << ", " << pqEntry.getParticleIndex() << ")";
  return os;
}

template<typename RandomIt, typename Compare = std::less<>>
void replace_heap(RandomIt first, RandomIt last, Compare comp = Compare())
{
  // From https://stackoverflow.com/questions/32672474/how-to-replace-top-element-of-heap-efficiently-withouth-re-establishing-heap-inv
  auto const size = last - first;
  if (size <= 1)
      return;
  typename std::iterator_traits<RandomIt>::difference_type k = 0;
  auto e = std::move(first[k]);
  auto const max_k = heap_parent(size - 1);
  while (k <= max_k) {
      auto max_child = heap_left(k);
      if (max_child < size - 1 && comp(first[max_child], first[max_child + 1]))
          ++max_child; // Go to right sibling.
      if (!comp(e, first[max_child]))
          break;
      first[k] = std::move(first[max_child]);
      k = max_child;
  }

  first[k] = std::move(e);
}


template<typename T>
class PriorityQueue {
  protected:
    std::vector<bool> particleIsInQueue;
    size_t maxSize;
    std::vector<std::unique_ptr<PQEntry<T>>> heap {};
    bool is_full = false;

  public:
    PriorityQueue(size_t maxSize, size_t numParticles) : maxSize(maxSize), particleIsInQueue(numParticles) {

    }

    PriorityQueue(const PriorityQueue&) = delete;
    PriorityQueue& operator=(const PriorityQueue&) = delete;


    bool push(T distanceSquared, npy_intp particleIndex, T ax, T ay, T az) {
      // Returns true if the particle was added to the queue, false if it was not
      if (contains(particleIndex)) return false;

      if(NPY_UNLIKELY(!is_full)) {
        heap.push_back(std::make_unique<PQEntry<T>>(distanceSquared, particleIndex, ax, ay, az));
        if (heap.size() == maxSize) {
          std::make_heap(heap.begin(), heap.end(), PQEntryPtrComparator<T>{});
          is_full = true;
        }
        particleIsInQueue[particleIndex] = true;
        return true;
      } else if (distanceSquared < topDistanceSquaredOrMax()) {

        particleIsInQueue[heap.front()->getParticleIndex()] = false;
        heap.front() = std::make_unique<PQEntry<T>>(distanceSquared, particleIndex, ax, ay, az);

        // using a custom replace_heap function is more efficient - in tests, around 5% faster for finding smoothing
        // than an STL-implemented pop-then-push approach.
        //
        // There is a minor danger that replace_heap uses a different sort of heap than std::make_heap, but in practice
        // this seems to work fine.

        replace_heap(heap.begin(), heap.end(), PQEntryPtrComparator<T>{});

        particleIsInQueue[particleIndex] = true;

        return true;

      } else {
        return false;
      }

    }

    bool push(T distanceSquared, npy_intp particleIndex) {
      return push(distanceSquared, particleIndex, 0.0, 0.0, 0.0);
    }

    bool contains(npy_intp particleIndex) const {
      return particleIsInQueue[particleIndex];
    }

    void checkConsistency(std::string context) const {
      // For debug
      bool errors_found = false;
      std::cerr << context;
      iterateHeapEntries([&](const PQEntry<T> & entry) {
        std::cerr << entry.getParticleIndex() << " ";
        if(!this->contains(entry.getParticleIndex())) {
          std::cerr << "<-";
          errors_found = true;
        }
      });

      if(errors_found) {
        exit(0);
      } else {
        std::cerr << std::endl;
      }

    }

    void updateDistances(std::function<void(PQEntry<T> &)> update_distance) {
      for(auto &entry : heap) {
        update_distance(*entry);
      }
      std::make_heap(heap.begin(), heap.end(), PQEntryPtrComparator<T>{});
    }

    void iterateHeapEntries(std::function<void(const PQEntry<T> &)> func) const {
      for(auto &entry : heap) {
        func(*entry);
      }
    }

    void pop() {
      particleIsInQueue[heap.front()->getParticleIndex()] = false;
      std::pop_heap(heap.begin(), heap.end(), PQEntryPtrComparator<T>{});
      heap.pop_back();
      is_full = false;
    }

    const PQEntry<T>& top() const {
      return *(heap.front());
    }

    inline T topDistanceSquaredOrMax() const {
      // Return the distance squared of the top element if the queue is full, otherwise return
      // the maximum value of the type (so that all attempts to push will succeed)
      if(NPY_LIKELY(is_full))
        return heap.front()->distanceSquared;
      else
        return std::numeric_limits<T>::max();
    }

    size_t size() const {
      return heap.size();
    }

    void clear() {
      for(auto &entry : heap) {
        this->particleIsInQueue[entry->getParticleIndex()] = 0;
      }
      heap.clear();
      is_full = false;
    }

    bool empty() const {
      return heap.empty();
    }

    inline bool full() const {
      return is_full;
    }

    size_t getMaxSize() const {
      return maxSize;
    }

};
