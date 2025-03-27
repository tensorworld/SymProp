#ifndef SYMPROP_FUNCTION_TIMER_HPP
#define SYMPROP_FUNCTION_TIMER_HPP

#include <chrono>
#include <string>

#ifdef FUNCTION_TIMER

class FunctionTimer {
  public:
  explicit FunctionTimer(std::string_view funcName);
  ~FunctionTimer();
  static void printTimes();
  static void clear();
  void stop();

  private:
  std::chrono::time_point<std::chrono::high_resolution_clock> m_start;
  std::string m_funcName;
  bool m_stopped = false;
};

#else

class FunctionTimer {
  public:
  explicit FunctionTimer([[maybe_unused]] std::string_view funcName) {};
  ~FunctionTimer() = default;
  static void printTimes() {}
  static void clear() {}
  void stop() {}
};

#endif

#define START_FUNCTION_TIMER() FunctionTimer __ft(__FUNCTION__);

#endif // LINBOOST_FUNCTION_TIMER_H