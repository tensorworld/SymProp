      
# SymProp: Tucker Decomposition for Sparse Symmetric Tensors

SymProp is a C++ software tool for computing the Tucker decomposition of high-order sparse symmetric tensors. It implements symmetry propagation to exploit symmetry throughout the intermediate computations (specifically in HOOI and HOQRI algorithms), leading to significant performance improvements and enabling analysis of higher-order tensors compared to existing methods.

## Prerequisites

*   **C++ Compiler:** A modern C++ compiler supporting C++20 standard (e.g., GCC >= 10, Clang >= 10).
*   **CMake:** Version 3.14 or higher.
*   **BLAS/LAPACK Library:** Either Intel MKL or OpenBLAS must be installed.

## Building SymProp

1.  **Clone the Repository:** catch2 will be cloned as a submodule.
    ```bash
    git clone --recursive https://github.com/tensorworld/SymProp.git
    cd SymProp
    ```

2.  **Configure with CMake:**
    Create a build directory and run CMake. You can specify build options here.
    ```bash
    mkdir build
    cd build
    cmake .. [options]
    ```

    **Common CMake Options:**
    *   `-DCMAKE_BUILD_TYPE=Release`: Build with optimizations (Recommended). Other options: `Debug`.
    *   `-DUSE_MKL=ON` (Default): Link against Intel MKL. Ensure MKL environment variables (MKL_ROOT) or paths are set correctly.
    *   `-DUSE_MKL=OFF`: Link against OpenBLAS. Ensure OpenBLAS is found by CMake.
    *   `-DBUILD_TESTS=ON` (Default): Build test executables.
    *   `-DBUILD_EXAMPLES=ON` (Default): Build the main `symtucker` example executable.
    *   `-DBUILD_BENCHMARKS=OFF` (Default): Build benchmark executables.
    *   `-DUSE_TIMER=ON` (Default): Enable internal function timers.
    *   `-DUSE_SANITIZER=OFF` (Default): Build with sanitizers (for debugging).

    *Example configuring for Release build using OpenBLAS:*
    ```bash
    cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_MKL=OFF
    ```

3.  **Compile:**
    ```bash
    make -j
    ```
    (Replace `-j` with the number of parallel jobs you want to use, e.g., `make -j4`)

## Running SymProp

The main executable `symtucker` will be located in the `build/examples/` directory (if `BUILD_EXAMPLES` was `ON`).

**Basic Syntax:**

```bash
./examples/symtucker -f <input_file> -r <rank> [options]
```

**Arguments:**

- `-f <path>`: (Required) Path to the input sparse symmetric tensor file (e.g., in FROSTT .tns format).

- `-r <rank>`: (Required) Target rank for the decomposition (default: 2).

- `-i <max_iter>`: Maximum number of iterations (default: 10).

- `-t <rel_tol>`: Relative tolerance for convergence check (default: 1e-6).

- `-d <delta>`: Regularization delta parameter (default: 0.5). Used for row-wise L2-norm regularization as described in Ke et al., 2019 [1]. Set to 0 to disable.

- `-a <alg>`: Algorithm to use:

    - 0: HOOI (Higher-Order Orthogonal Iteration)

    - 1: HOQRI (Higher-Order QR Iteration)

    - 2: AltHOOI (Alternative HOOI using Eigen-decomposition, potentially faster but may have stability issues)

    - -1 (Default): Run and report results for all algorithms (HOOI, HOQRI, AltHOOI).

- `-s <seed>`: Random seed for initialization (default: 0).

- `-o <output_file>`: Path to save the output factors (optional).

- `-init <method>`: Initialization method:

    - rnd (Default): Random initialization.

    - hosvd: Higher-Order SVD (Requires MKL which supports sparse SVD, i.e., USE_MKL=ON).

    - hoevd: Higher-Order Eigenvalue Decomposition.

**Example Usage:**

Assuming you are in the build directory and using an example tensor file located at `../examples/contact-5.tns`:

```bash
./examples/symtucker -f ../examples/contact-5.tns -r 8 -i 20 -a 0 -d 0.1 -o contact-5-output.txt
```

This command runs the HOOI algorithm (-a 0) for 20 iterations (-i 20) to compute a rank-8 (-r 8) decomposition of contact-5.tns, using a regularization delta of 0.1 (-d 0.1), and saves the result to contact-5-output.txt.

Example tensor files can be found in the examples/ directory of the source tree.

**Output Format:**

If an output file (-o) is specified, it will contain:

1. Line 1: Mode-1 matricized C:

2. Lines 2 to rank+1: Comma-separated rows of the mode-1 unfolding (matricization) of the core tensor C.

3. Line rank+2: Factor U:

4. Following lines: Comma-separated rows of the factor matrix U (which is the same for all modes due to symmetry). The number of rows corresponds to the dimension size of the input tensor.

## Testing

If built with `-DBUILD_TESTS=ON` (the default), test executables will be created in the `build/tests/` directory.

To run the tests, navigate to the test directory and execute the binaries individually:

```bash
cd build/tests
./test_compressed_dtree  
./test_dtree 
# ... and so on
```

*Note: Test coverage is currently partial.*

SymProp is licensed under the MIT License.

[1] Z. T. Ke, F. Shi, and D. Xia, “Community detection for hypergraph networks via regularized tensor power iteration,” arXiv: Methodology, 2019. [Online]. Available: https://api.semanticscholar.org/CorpusID:202577680