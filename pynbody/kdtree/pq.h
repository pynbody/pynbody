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
  bool operator()(PQEntry<T>* lhs, PQEntry<T>* rhs) {
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
    std::vector<PQEntry<T>*> heap {};
    bool is_full = false;

  public:
    PriorityQueue(size_t maxSize, size_t numParticles) : maxSize(maxSize), particleIsInQueue(numParticles) {

    }

    // no copying allowed
    PriorityQueue(const PriorityQueue&) = delete;
    PriorityQueue& operator=(const PriorityQueue&) = delete;


    bool push(T distanceSquared, npy_intp particleIndex, T ax, T ay, T az) {
      // Returns true if the particle was added to the queue, false if it was not
      if (contains(particleIndex)) return false;

      if(NPY_UNLIKELY(!is_full)) {
        heap.push_back(new PQEntry<T>(distanceSquared, particleIndex, ax, ay, az));
        if (heap.size() == maxSize) {
          std::make_heap(heap.begin(), heap.end(), PQEntryPtrComparator<T>{});
          is_full = true;
        }
        particleIsInQueue[particleIndex] = true;
        return true;
      } else if (distanceSquared < topDistanceSquared()) {
        /*
        pop();
        heap.push_back(new PQEntry<T>(distanceSquared, particleIndex, ax, ay, az));
        std::push_heap(heap.begin(), heap.end(), PQEntryPtrComparator<T>{});
        particleIsInQueue[particleIndex] = true;
        is_full = true; */
        
        auto heap_front = heap.front();
        particleIsInQueue[heap_front->getParticleIndex()] = false;
        delete heap_front;
        heap.front() = new PQEntry<T>(distanceSquared, particleIndex, ax, ay, az);
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
      delete heap.back();
      heap.pop_back();
      is_full = false;
    }

    const PQEntry<T>& top() const {
      return *(heap.front());
    }

    inline T topDistanceSquared() const {
      return heap.front()->distanceSquared;
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
      for(auto &entry : heap) {
        this->particleIsInQueue[entry->getParticleIndex()] = 0;
        delete entry;
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