#ifndef SYMPROP_DISPATCHER_HPP
#define SYMPROP_DISPATCHER_HPP

#include <array>
#include <functional>
#include <memory>
#include <stdexcept>

#include <type_traits>

template <typename> struct function_traits;

template <typename T>
struct function_traits : public function_traits<decltype(&T::operator())> {};

template <typename C, typename R, typename... Args>
struct function_traits<R (C::*)(Args...) const> {
  using return_type = R;
  using args_tuple = std::tuple<Args...>;
};

template <template <size_t> class T, size_t MinN, size_t MaxN>
class TemplateDispatcher {
private:
  using Traits = function_traits<T<MinN>>;
  using R = typename Traits::return_type;
  using ArgsTuple = typename Traits::args_tuple;

  template <typename Tuple> struct to_function;
  template <typename... Args> struct to_function<std::tuple<Args...>> {
    using type = std::function<R(Args...)>;
  };

  using FuncType = typename to_function<ArgsTuple>::type;
  std::vector<FuncType> dispatch_table;
  size_t min_n;

public:
  TemplateDispatcher() : dispatch_table(MaxN - MinN + 1), min_n(MinN) {
    [this]<size_t... Is>(std::index_sequence<Is...>) {
      ((dispatch_table[Is] = [](auto &&...args) -> R {
         return T<Is + MinN>{}(std::forward<decltype(args)>(args)...);
       }),
       ...);
    }(std::make_index_sequence<MaxN - MinN + 1>{});
  }

  template <typename... Args> R operator()(size_t N, Args &&...args) const {
    if (N >= min_n && N < min_n + dispatch_table.size()) {
      return dispatch_table[N - min_n](std::forward<Args>(args)...);
    } else {
      throw std::out_of_range("N is out of the valid range");
    }
  }
};

template <template <size_t> class T, size_t MinN, size_t MaxN>
auto make_dispatcher() {
  return TemplateDispatcher<T, MinN, MaxN>();
}

#endif // SYMPROP_DISPATCHER_HPP
