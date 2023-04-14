# CuTe

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://YichengDWu.github.io/CuTe.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://YichengDWu.github.io/CuTe.jl/dev/)
[![Build Status](https://github.com/YichengDWu/CuTe.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/YichengDWu/CuTe.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/YichengDWu/CuTe.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/YichengDWu/CuTe.jl)

Please refer to NVIDIA's [CuTe](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/00_quickstart.md) for tutorials.

## Layout
### Constructing a `Layout`

```julia
layout_2x4 = make_layout((2,(2,2)),(4,(1,2)))
print("Shape:", shape(layout_2x4), "\n"
      "Stride:", stride(layout_2x4), "\n"
      "Size:", size(layout_2x4), "\n"
      "Rank:", rank(layout_2x4), "\n"
      "Depth:", depth(layout_2x4), "\n"
      "Cosize:", cosize(layout_2x4))

```