#ifndef SYMPROP_THREADING_HPP
#define SYMPROP_THREADING_HPP

#include <omp.h>
#include <algorithm>

namespace symprop {

namespace threading {

constexpr std::pair<int, int> get_range(int work, int nworker, int worker_id) {
  int base = work / nworker;
  int remainder = work % nworker;

  // Calculate the starting point of the range
  int begin = std::max(0, worker_id * base + std::min(worker_id, remainder));
  // Calculate the end point of the range
  int end = begin + base + (worker_id < remainder);

  return std::make_pair(begin, end);
}

inline size_t nthreads() { return static_cast<size_t>(omp_get_max_threads()); }
inline size_t thread_id() { return static_cast<size_t>(omp_get_thread_num()); }

} // namespace threading

} // namespace symprop

#endif