# SkeAI

SkeAI is an experimental artificial intelligence project built from scratch.

## Current stage

**SkeAI 0.4 — C++ engine foundation**

The model is intentionally small. This milestone is about building the numerical engine correctly before scaling the model.

The foundation includes:

- A character-level Arabic/English tokenizer.
- A contiguous tensor abstraction backed by C++-owned storage.
- C++ implementations of tensor operations, neural layers, losses, and SGD.
- A sequential neural network.
- A tiny fixed-context character language model.
- JSON checkpoints and resume training.
- Training telemetry and repeatable benchmarks.
- Automated tests through GitHub Actions.

## Build and test

From the repository root, the Makefile provides the same entry points on Linux and Termux:

    make check-sources
    make build
    make test
    make info
    make clean

- `make check-sources` discovers Python, C/C++, header, and assembly source files and checks Python syntax.
- `make build` builds the native C++ extension and byte-compiles Python modules.
- `make test` runs source checks, builds the extension, and runs the full unittest suite.
- `make info` lists recognized source files.
- `make clean` removes generated Python bytecode and the local native extension.

The native extension is built by:

    python -m tools.build_cpp

It creates a platform-specific _cpp extension inside src/skeai/.

### Termux setup

Install the available toolchain packages:

    pkg update
    pkg install python clang make git

Then run:

    make test

If the native build reports that the Python library or headers are missing, keep the full error output; Termux's Python packaging differs from desktop Linux and the build script may need a platform-specific adjustment.

Python remains responsible for high-level model orchestration and the public API. C++ owns the scalar tensor storage and executes the numerical kernels.

## Run training

    python -m training.train_tiny

Resume from the checkpoint:

    python -m training.train_tiny --resume

## Benchmark

    python -m benchmarks.benchmark_engine

The benchmark measures matrix multiplication, Dense forward/backward, a real training step, layer timing, and parameter count.

The complete training step is the primary performance signal. An isolated kernel improvement is not accepted when the end-to-end workload gets slower.

## Tests

Build the C++ engine first:

    python -m tools.build_cpp
    python -m unittest discover -s tests -v

GitHub Actions performs the same build before running the tests.

## Architecture

    Python Runtime
          |
      Tensor API
          |
    Python/C++ Boundary
          |
    +---------------------------+
    |       SkeAI C++ Core      |
    |---------------------------|
    | C++ tensor storage        |
    | MatMul                    |
    | Dense                     |
    | ReLU / Tanh               |
    | MSE / CrossEntropy        |
    | SGD                       |
    +-------------+-------------+
                  |
                 CPU

The important boundary is the tensor storage itself:

    Old experimental path:
    Python nested lists -> native C kernels

    New foundation:
    Python Tensor metadata
            |
            v
    C++-owned contiguous buffer
            |
            v
    C++ numerical kernels
            |
            v
           CPU

The old C bridge is removed instead of being retained as a compatibility layer.

## Design rules

1. Python handles orchestration, model structure, serialization, and user-facing APIs.
2. C++ owns tensor scalar storage.
3. Numerical kernels operate directly on contiguous native memory.
4. Heavy native kernels release the Python GIL while computing.
5. Native behavior is validated through end-to-end Python tests.
6. Performance decisions are based on complete training-step measurements.

The project does not depend on PyTorch, TensorFlow, NumPy, or another ML framework.

## Long-term direction

The same C++ core is intended to grow toward:

    Tensor and memory runtime
            |
    Neural-network runtime
            |
    Model and memory system
            |
    Task orchestration
       /            \
    PC/OS tools    Voice I/O
            |
          SkeOS

This is a future direction. The current project is still a small learning and engineering system.

## Philosophy

SkeAI is built to understand the internals instead of hiding them behind a framework.

The numerical engine is therefore built as a real native C++ core from the foundation, not as a thin optimization layer placed over Python container objects.
