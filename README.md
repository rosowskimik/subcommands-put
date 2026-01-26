# ExecuTorch Speechcommands C++ Project

A simple application using PyTorch/ExecuTorch to detect predefined speechcommands from audio  input.
This repository contains Python code used to train and export quantized model and sources to build
a C++ application used for inference on that model.

# Building the app

1. Fetch this repository (with submodules)

```sh
git clone --recurse-submodules https://github.com/rosowskimik/subcommands-put.git
```

2. Setup Python dependencies (venv recommended)

```sh
pip install -r python/requirements.txt
```

3. Configure CMake build

```sh
cmake -S. -Bbuild
```

4. Build the application

```sh
cmake --build build -j
```

The resulting binary should be available under `build/commands`

# Running the app

## Application arguments

```sh
❯ commands --help
Project Executorch
Usage:
  commands [OPTION...]

  -m, --model arg           Path to model (default: python/tiny.pte)
  -i, --input arg           Path to raw mel data
  -b, --bench [=arg(=500)]  Run benchmark
  -w, --warmup [=arg(=50)]  Warmup runs before benchmark (default: 0)
  -h, --help                Print usage
```

## Inference

```sh
commands --model <path_to_model.pte> --input <path_to_mel_data>
```

## Benchmark

```sh
commands --model <path_to_model.pte> --input <path_to_mel_data> --bench --warmup
```
