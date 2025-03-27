#include <utils/function_timer.hpp>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <unordered_map>
#include <vector>

#ifdef FUNCTION_TIMER

class FunctionTimeAccumulator {
public:
  static FunctionTimeAccumulator &getInstance() {
    static FunctionTimeAccumulator instance;
    return instance;
  }

  void addTime(const std::string &funcName, long long time) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_accumulatedTime[funcName] += time;
    m_numCalls[funcName]++;
  }

  void printTimes() {
    std::lock_guard<std::mutex> lock(m_mutex);

    // Convert to vector and sort
    std::vector<std::pair<std::string, long long>> sorted_times(
        m_accumulatedTime.begin(), m_accumulatedTime.end());
    std::sort(sorted_times.begin(), sorted_times.end(),
              [](const auto &a, const auto &b) { return b.second < a.second; });

    // Print sorted times
    std::cout << "Accumulated function times:\n";
    for (const auto &entry : sorted_times) {
      // Convert microseconds to milliseconds and format the output
      std::streamsize ss = std::cout.precision();
      double time_in_ms = static_cast<double>(entry.second) / 1000.0;
      std::cout << std::fixed << std::setprecision(3) << std::setw(10)
                << time_in_ms << " ms  | " << std::setw(10)
                << m_numCalls[entry.first] << " calls | " << entry.first
                << "\n";
      std::cout.precision(ss);
    }
  }

  void clear() {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_accumulatedTime.clear();
    m_numCalls.clear();
  }

private:
  std::unordered_map<std::string, long long> m_accumulatedTime;
  std::unordered_map<std::string, long long> m_numCalls;
  std::mutex m_mutex;
};

FunctionTimer::FunctionTimer(const std::string_view funcName)
    : m_start(std::chrono::high_resolution_clock::now()), m_funcName(funcName) {
}

FunctionTimer::~FunctionTimer() { stop(); }

void FunctionTimer::printTimes() {
  FunctionTimeAccumulator::getInstance().printTimes();
}
void FunctionTimer::clear() { FunctionTimeAccumulator::getInstance().clear(); }

void FunctionTimer::stop() {
  if (m_stopped) {
    return;
  }
  m_stopped = true;
  auto m_stop = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(m_stop - m_start);
  FunctionTimeAccumulator::getInstance().addTime(m_funcName, duration.count());
}

#endif // FUNCTION_TIMER