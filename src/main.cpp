#include "torch_consts_generated.h"

#include <algorithm>
#include <cxxopts.hpp>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>

#include <chrono>
#include <cstdint>
#include <format>
#include <fstream>
#include <ios>
#include <iostream>
#include <stdexcept>

using namespace ::executorch::aten;
using namespace ::executorch::extension;

constexpr auto NUM_ELEMENTS = N_FRAMES * N_BANKS;
constexpr auto EXPECTED_BYTES = NUM_ELEMENTS * sizeof(float);

static float input_data[NUM_ELEMENTS];

template <class T> inline void _do_not_optimize(T &&value) {
  asm volatile("" ::"g"(value) : "memory");
}

TensorPtr load_data(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    throw std::runtime_error(std::format("Cannot open '{}'\n", path));
  }
  f.seekg(0, std::ios::end);
  const std::streamoff size = f.tellg();
  f.seekg(0, std::ios::beg);

  if (static_cast<size_t>(size) != EXPECTED_BYTES) {
    throw std::runtime_error(
        std::format("Data file malformed (expected {}, got {})\n",
                    EXPECTED_BYTES, static_cast<size_t>(size)));
  }

  f.read(reinterpret_cast<char *>(input_data),
         static_cast<std::streamsize>(EXPECTED_BYTES));
  if (!f) {
    throw std::runtime_error(std::format("Failed to read file '{}'\n", path));
  }

  return from_blob(input_data, {1, N_BANKS, N_FRAMES});
}

template <bool bench = true> int run(Module &module, TensorPtr data) {
  if constexpr (!bench) {
    std::cout << std::format("Running inference\n");
  }

  auto result = module.forward(data);
  if (!result.ok()) {
    throw std::runtime_error(std::format("Inference failed\n"));
  }
  auto tensor = result->data()->toTensor();

  const float *out_data = tensor.const_data_ptr<float>();
  const int numel = tensor.numel();

  int max_idx = 0;
  float max_val = out_data[0];
  for (int i = 1; i < numel; ++i) {
    if (out_data[i] > max_val) {
      max_val = out_data[i];
      max_idx = i;
    }
  }

  if constexpr (!bench) {
    std::cout << std::format("Predicted command: '{}' (label idx {})\n",
                             LABELS[max_idx], max_idx);
  }

  return max_idx;
}

void benchmark(Module &module, TensorPtr data, uint32_t runs, uint32_t warmup) {
  using clock = std::chrono::steady_clock;

  if (runs == 0) {
    throw std::runtime_error("Can't benchmark with 0 runs!\n");
  }

  if (warmup > 0) {
    std::cout << std::format("Running {} warmup rounds\n", warmup);
  }
  for (std::uint32_t i = 0; i < warmup; ++i) {
    auto r = run(module, data);
    _do_not_optimize(r);
  }

  std::chrono::nanoseconds min_dur = std::chrono::nanoseconds::max();
  std::chrono::nanoseconds max_dur = std::chrono::nanoseconds::zero();
  std::uint64_t sum_ns = 0;

  std::cout << std::format("Running inference {} times\n", runs);
  for (std::uint32_t i = 0; i < runs; ++i) {
    const auto t0 = clock::now();

    auto r = run(module, data);
    _do_not_optimize(r);

    const auto t1 = clock::now();
    const auto dur =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0);

    min_dur = std::min(min_dur, dur);
    max_dur = std::max(max_dur, dur);
    sum_ns += static_cast<std::uint64_t>(dur.count());
  }

  std::cout << std::format(
      "Bench results:\nmin: {}ns, max: {}ns, avg: {}ns\n", min_dur.count(),
      max_dur.count(),
      std::chrono::nanoseconds(static_cast<std::uint64_t>(sum_ns / runs))
          .count());
}

cxxopts::ParseResult parse_args(int argc, char **argv) {
  cxxopts::Options options("commands", "Project Executorch");
  // clang-format off
  options.add_options()
    ("m,model", "Path to model", cxxopts::value<std::string>()->default_value("python/tiny_qt.pte"))
    ("i,input", "Path to raw mel data", cxxopts::value<std::string>())
    ("b,bench", "Run benchmark", cxxopts::value<uint32_t>()->implicit_value("500"))
    ("w,warmup", "Warmup runs before benchmark", cxxopts::value<uint32_t>()->implicit_value("50")->default_value("0"))
    ("h,help", "Print usage");
  // clang-format on

  cxxopts::ParseResult result = options.parse(argc, argv);

  if (result.count("help")) {
    std::cout << options.help() << std::endl;
    exit(0);
  }

  return result;
}

int main(int argc, char **argv) {
  auto args = parse_args(argc, argv);

  auto module = Module(args["model"].as<std::string>());
  auto data = load_data(args["input"].as<std::string>());

  if (args.count("bench") == 0) {
    run<false>(module, data);
    return 0;
  }

  benchmark(module, data, args["bench"].as<uint32_t>(),
            args["warmup"].as<uint32_t>());
  return 0;
}
