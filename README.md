# MoYe

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://YichengDWu.github.io/MoYe.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://YichengDWu.github.io/MoYe.jl/dev/)
[![Build Status](https://github.com/YichengDWu/MoYe.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/YichengDWu/MoYe.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/YichengDWu/MoYe.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/YichengDWu/MoYe.jl)

`MoYe.jl` is NVIDIA's [Cutlass/CuTe](https://github.com/NVIDIA/cutlass/blob/main/) implemented in Julia.
The primary purpose of developing this library is my desire to learn CuTe.

The name **Mo Ye** is derived from an ancient Chinese [legend of swordsmiths](https://en.wikipedia.org/wiki/Gan_Jiang_and_Mo_Ye).

The documentation is mostly my learning notes. Please refer to CuTe's documentation for more details.

GEMM essentially faces two main performance hurdles not implemented yet:

    1. Swizzling to prevent bank conflicts.
    2. An efficient epilogue, which involves transferring data from registers to shared memory, followed by a vectorized transfer back to global memory.

Since I've sold my computer, I no longer have access to an NVIDIA GPU, thus the development of this library will be put on hold indefinitely.
