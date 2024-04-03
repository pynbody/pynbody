#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>
#include <limits>
#include <memory>

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

bool operator<(const std::unique_ptr<PQEntry<float>>& lhs, const std::unique_ptr<PQEntry<float>>& rhs) {
  // It's unclear to me why this should be necessary, since std::unique_ptr comparisons should map
  // onto underlying comparisons according to https://en.cppreference.com/w/cpp/memory/unique_ptr/operator_cmp
  // But wihtout it, the comparisons seem to be wrong.
  return lhs->distanceSquared < rhs->distanceSquared;
}

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
    std::vector<std::unique_ptr<PQEntry<T>>> heap {};

  public:
    PriorityQueue(size_t maxSize, size_t numParticles) : maxSize(maxSize), particleIsInQueue(numParticles) {

    }

    // no copying allowed
    PriorityQueue(const PriorityQueue&) = delete;
    PriorityQueue& operator=(const PriorityQueue&) = delete;


    inline void push(T distanceSquared, npy_intp particleIndex, T ax, T ay, T az) {
      if (contains(particleIndex)) return;

      if (distanceSquared < topDistanceSquaredOrMax()) {
        if(full()) pop();

        heap.push_back(std::make_unique<PQEntry<T>>(distanceSquared, particleIndex, ax, ay, az));
        std::push_heap(heap.begin(), heap.end());
        particleIsInQueue[particleIndex] = true;

      } 
      
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
      std::make_heap(heap.begin(), heap.end());
    }

    void iterateHeapEntries(std::function<void(const PQEntry<T> &)> func) const {
      for(auto &entry : heap) {
        func(*entry);
      }
    }

    void push(T distanceSquared, npy_intp particleIndex) {
      push(distanceSquared, particleIndex, 0.0, 0.0, 0.0);
    }

    void pop() {
      particleIsInQueue[heap.front()->getParticleIndex()] = false;
      std::pop_heap(heap.begin(), heap.end());
      heap.pop_back();
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