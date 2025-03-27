#include "utils/function_timer.hpp"
#include <utils/types.hpp>
#include <detail/sparse/spsymtensor.hpp>
#include <detail/dense/matrixops.hpp>

#include <iomanip>
#include <chrono>
#include <fstream>
#include <random>

using namespace symprop;

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <input_file> <R>" << std::endl;
    return 1;
  }

  std::string input_file = argv[1];
  size_t R = std::stoi(argv[2]);

  std::vector<dim_t> indices;
  std::vector<real_t> vals;

  std::ifstream file(input_file);
  if (!file.is_open()) {
    std::cerr << "Error opening file: " << input_file << std::endl;
    return 1;
  }

  order_t order;
  size_t dim;
  file >> order >> dim;
  std::vector<dim_t> nnz_index;

  while (true) {
    nnz_index.clear();
    for (order_t i = 0; i < order; i++) {
      dim_t val;
      file >> val;
      nnz_index.push_back(val);
    }

    real_t val;
    file >> val;
    if (!file.good())
      break;
    std::sort(nnz_index.begin(), nnz_index.end());
    vals.push_back(val);
    indices.insert(indices.end(), nnz_index.begin(), nnz_index.end());
  }

  file.close();

  std::vector<real_t> factor(dim * R);
  // random initialization
  std::mt19937 gen(1);
  std::uniform_real_distribution<real_t> dist(1.0, 2.0);
  for (size_t i = 0; i < dim * R; i++) {
    factor[i] = dist(gen);
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  SpSymtensor symtens(dim, order, indices, vals);
  auto t2 = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  std::cout << "Insertion time: " << duration << " ms" << std::endl;

  size_t ncols = symtens_size(R, order - 1);
  std::vector<real_t> core(R * ncols);
  std::vector<real_t> core_prev(R * ncols);
  std::vector<real_t> Y(dim * ncols);
  std::vector<real_t> coeff = symtens_coeff(R, order - 1);

  for (size_t i = 0; i < 10; i++) {  
    t1 = std::chrono::high_resolution_clock::now();
    symtens.S3TTMc(Y, factor, R, true);
    matrix::gemmtn(core, factor, Y, dim, R, ncols);
    matrix::symgemm(factor, Y, core, coeff, dim, R);
    t2 = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    std::cout << "S3TTMcTC Time: " << duration << " ms" << std::endl;
  }

  FunctionTimer::printTimes();
  // std::ofstream out_file("ttmc.out");
  // out_file << std::fixed;
  // out_file << std::setprecision(5);
  // size_t dim1 = symtens_size(R, order - 1);
  // for (size_t i = 0; i < dim; i++) {
  //   for (size_t j = 0; j < dim1; j++) {
  //     out_file << res[i * dim1 + j] << ", ";
  //   }
  //   out_file << "\n";
  // }
  // out_file.close();

  return 0;
}
