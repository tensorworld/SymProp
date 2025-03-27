#ifndef SYMPROP_BUFFER2D_HPP
#define SYMPROP_BUFFER2D_HPP

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <new>
#include <span>
#include <stdexcept>
#include <type_traits>

namespace symprop {

template <typename T> class Buffer2D {
  static_assert(std::is_trivial_v<T>, "T must be a trivial type");

private:
  T *data;
  size_t rows;
  size_t cols;
  size_t capacity;

public:
  Buffer2D() : data(nullptr), rows(0), cols(0), capacity(0) {}

  Buffer2D(size_t numRows, size_t numCols)
      : rows(numRows), cols(numCols), capacity(numRows * numCols) {
    data = static_cast<T *>(std::malloc(capacity * sizeof(T)));
    if (!data)
      throw std::bad_alloc();
  }

  ~Buffer2D() {
    if (data)
      std::free(data);
  }

  // avoid copying
  Buffer2D(const Buffer2D &other) = delete;
  Buffer2D &operator=(const Buffer2D &other) = delete;

  Buffer2D(Buffer2D &&other) noexcept
      : data(other.data), rows(other.rows), cols(other.cols),
        capacity(other.capacity) {
    other.data = nullptr;
    other.rows = other.cols = other.capacity = 0;
  }

  Buffer2D &operator=(Buffer2D &&other) noexcept {
    if (this != &other) {
      if (data)
        std::free(data);
      data = other.data;
      rows = other.rows;
      cols = other.cols;
      capacity = other.capacity;
      other.data = nullptr;
      other.rows = other.cols = other.capacity = 0;
    }
    return *this;
  }

  void swap(Buffer2D &other) noexcept {
    std::swap(data, other.data);
    std::swap(rows, other.rows);
    std::swap(cols, other.cols);
    std::swap(capacity, other.capacity);
  }

  void reshape(size_t new_rows, size_t new_cols) {
    size_t newCapacity = new_rows * new_cols;
    assert(newCapacity <= capacity);
    if (newCapacity > capacity) {
      T *newData = static_cast<T *>(std::malloc(newCapacity * sizeof(T)));
      if (!newData)
        throw std::bad_alloc();
      if (data)
        std::free(data);
      data = newData;
      capacity = newCapacity;
    }
    rows = new_rows;
    cols = new_cols;
  }

  T &operator()(size_t row, size_t col) { return data[row * cols + col]; }

  const T &operator()(size_t row, size_t col) const {
    return data[row * cols + col];
  }

  std::span<T> operator[](size_t row) {
    return std::span<T>(data + row * cols, cols);
  }

  std::span<const T> operator[](size_t row) const {
    return std::span<const T>(data + row * cols, cols);
  }

  size_t numRows() const { return rows; }
  size_t numCols() const { return cols; }
  size_t size() const { return rows * cols; }

  // Conversion to std::span for the entire buffer
  operator std::span<T>() { return std::span<T>(data, rows * cols); }
  operator std::span<const T>() const {
    return std::span<const T>(data, rows * cols);
  }
};

template <typename T> void swap(Buffer2D<T> &lhs, Buffer2D<T> &rhs) noexcept {
  lhs.swap(rhs);
}

} // namespace symprop

#endif // SYMPROP_MATRIX2D_HPP