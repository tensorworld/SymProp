#include <detail/sparse/spsymtensor.hpp>
#include <detail/tucker.hpp>
#include <utils/function_timer.hpp>
#include <utils/loader.hpp>
#include <utils/types.hpp>

#include <chrono>
#include <fstream>
#include <functional>
#include <iomanip>
#include <random>

using namespace symprop;

template <typename Alg>
SymTuckerDecomp run_algorithm(Alg alg, const Config &con, const std::string &name) {
  std::cout << "Running " << name << " algorithm" << std::endl;
  auto t1 = std::chrono::high_resolution_clock::now();
  auto tucker = alg(con);
  auto t2 = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  std::cout << "---------- Time: " << duration << " ms ----------" << std::endl;
  FunctionTimer::printTimes();
  FunctionTimer::clear();
  return tucker;
}

void help(const char *prog) {
  std::cout << prog
            << " -f <input_file> -i <max_iter> -r <rank> -t <rel_tol> -d "
               "<delta> -a <alg> -s <seed> -o <output_file> -init <init method>"
            << std::endl;
}

struct Input {
  Config config;
  std::string filePath;
  std::string outputFile;
  int alg = -1;
};

std::optional<Input> parse(int argc, char **argv) {
  Input input;
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "-f" && i + 1 < argc) {
      input.filePath = argv[i + 1];
      i++;
    } else if (arg == "-i" && i + 1 < argc) {
      input.config.max_iter = std::stod(argv[i + 1]);
      i++;
    } else if (arg == "-r" && i + 1 < argc) {
      input.config.rank = std::stoi(argv[i + 1]);
      i++;
    } else if (arg == "-t" && i + 1 < argc) {
      input.config.rel_tol = std::stod(argv[i + 1]);
      i++;
    } else if (arg == "-d" && i + 1 < argc) {
      input.config.delta = std::stod(argv[i + 1]);
      i++;
    } else if (arg == "-a" && i + 1 < argc) {
      input.alg = std::stoi(argv[i + 1]);
      i++;
    } else if (arg == "-s" && i + 1 < argc) {
      input.config.seed = std::stoi(argv[i + 1]);
      i++;
    } else if (arg == "-o" && i + 1 < argc) {
      input.outputFile = argv[i + 1];
      i++;
    } else if (arg == "-init" && i + 1 < argc) {
      input.config.init = argv[i + 1];
      i++;
    } else if (arg == "-h") {
      help(argv[0]);
      return std::nullopt;
    } else {
      std::cout << "Option " << argv[i] << " is not defined\n";
      help(argv[0]);
      return std::nullopt;
    }
  }
  return input;
}

int main(int argc, char **argv) {
  auto input = parse(argc, argv);
  if (!input.has_value()) {
    return 1;
  }

  SymtensLoader loader(input->filePath);
  auto t1 = std::chrono::high_resolution_clock::now();
  SpSymtensor symtens(loader.dim(), loader.order(), loader.unnz_inds(),
                      loader.unnzs());
  auto t2 = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  std::cout << "Insertion time: " << duration << " ms" << std::endl;

  FunctionTimer::printTimes();
  FunctionTimer::clear();

  std::string outpath = input->outputFile;
  if (outpath.empty()) {
    outpath = "/dev/null";
  } else if (outpath == "stdout") {
    outpath = "/dev/stdout";
  } else if (outpath == "stderr") {
    outpath = "/dev/stderr";
  }

  std::ofstream out(outpath); 

  if (input->alg == -1) {
    run_algorithm([&symtens](const Config &c) { return symtens.TuckerHOOI(c); },
                  input->config, "HOOI").print(out);
    run_algorithm(
        [&symtens](const Config &c) { return symtens.TuckerHOQRI(c); },
        input->config, "HOQRI").print(out);
    run_algorithm(
        [&symtens](const Config &c) { return symtens.TuckerAltHOOI(c); },
        input->config, "AltHOOI").print(out);
    return 0;
  } else if (input->alg == 0) {
    run_algorithm([&symtens](const Config &c) { return symtens.TuckerHOOI(c); },
                  input->config, "HOOI").print(out);
  } else if (input->alg == 1) {
    run_algorithm(
        [&symtens](const Config &c) { return symtens.TuckerHOQRI(c); },
        input->config, "HOQRI").print(out);
  } else if (input->alg == 2) {
    run_algorithm(
        [&symtens](const Config &c) { return symtens.TuckerAltHOOI(c); },
        input->config, "AltHOOI").print(out);
  }

  return 0;
}
