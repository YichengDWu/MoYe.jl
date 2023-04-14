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
print("Shape: ", shape(layout_2x4)) # (2, (2, 2))
print("Stride: ", stride(layout_2x4)) # (4, (1, 2))
print("Size: ", size(layout_2x4)) # 8 
print("Rank: ", rank(layout_2x4)) #2
print("Depth: ", depth(layout_2x4)) # 2
print("Cosize: ", cosize(layout_2x4)) # 8
```

### Flatten
```julia
layout = make_layout(((4,3), 1), ((3, 1), 0))
print(flatten(layout)) # (4, 3, 1):(3, 1, 0)
```

### Coalesce

```julia
layout = make_layout((2,(1,6)), (1,(6,2)))
print(coalesce(layout)) # 12:1
```

### Composition
```julia
make_layout(20,2) ∘ make_layout((4,5),(1,4)) # (4, 5):(2, 8)
```

### Complement

### Product

### Division