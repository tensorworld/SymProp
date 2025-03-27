cmake_minimum_required(VERSION 3.14)
project(SymProp VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
cmake_policy(SET CMP0060 NEW)

# Options
option(BUILD_TESTS "Build tests" ON)
option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_BENCHMARKS "Build benchmarks" OFF)
option(USE_TIMER "Enable function timer" ON)
option(USE_SANITIZER "Build with sanitizer" OFF)
option(USE_MKL "Use Intel MKL" ON)

# Set default build type to Release
if(NOT CMAKE_BUILD_TYPE)
 set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")

# Sanitizer options
if(USE_SANITIZER)
 set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address,undefined -fno-omit-frame-pointer")
 set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address,undefined")
endif()

include_directories(include)

find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# Find dependencies
if(USE_MKL)
 find_package(MKL REQUIRED)
 message(STATUS "MKL include dirs: ${MKL_ROOT}/include")
 include_directories(${MKL_ROOT}/include)
 add_compile_definitions(USE_MKL)
else()
 find_package(OpenBLAS REQUIRED)
 include_directories(${OpenBLAS_INCLUDE_DIRS})
endif()

if(USE_TIMER)
 add_compile_definitions(FUNCTION_TIMER)
endif()

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
set(MATH_LIBS ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

add_library(${PROJECT_NAME} STATIC
 src/dense/symtensor.cpp
 src/dense/symtensor_array.cpp
 src/dense/matrixops.cpp
 src/sparse/spsymtensor.cpp
 src/sparse/matrixops.cpp
 src/utils/function_timer.cpp
 src/utils/loader.cpp
 src/dtree.cpp
 src/tucker.cpp
 src/compressed_dtree.cpp
)

target_link_libraries(${PROJECT_NAME} PUBLIC ${MATH_LIBS})


# Examples
if(BUILD_EXAMPLES)
 add_subdirectory(examples)
endif()

# Tests
if(BUILD_TESTS)
 add_subdirectory(extern/Catch2)
 add_subdirectory(tests)
endif()

# Benchmarks
if(BUILD_BENCHMARKS)
 add_subdirectory(bench)
endif()