# CuTe

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://YichengDWu.github.io/CuTe.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://YichengDWu.github.io/CuTe.jl/dev/)
[![Build Status](https://github.com/YichengDWu/CuTe.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/YichengDWu/CuTe.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/YichengDWu/CuTe.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/YichengDWu/CuTe.jl)

Please refer to NVIDIA's [CuTe](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/00_quickstart.md) for tutorials.

## Layout
### Constructing a `Layout`

```julia
julia> print("Shape: ", shape(layout_2x4))
Shape: (2, (2, 2))
julia> print("Stride: ", stride(layout_2x4))
Stride: (4, (1, 2))
julia> print("Size: ", size(layout_2x4))
Size: 8
julia> print("Rank: ", rank(layout_2x4))
Rank: 2
julia> print("Depth: ", depth(layout_2x4))
Depth: 2
julia> print("Cosize: ", cosize(layout_2x4))
Cosize: 8
julia> print_layout(layout_2x4)
(2, (2, 2)):(4, (1, 2))
      1   2   3   4
    +---+---+---+---+
 1  | 1 | 2 | 3 | 4 |
    +---+---+---+---+
 2  | 5 | 6 | 7 | 8 |
    +---+---+---+---+
```

### Flatten
```julia
julia> layout = make_layout(((4,3), 1), ((3, 1), 0))
((4, 3), 1):((3, 1), 0)

julia> print(flatten(layout))
(4, 3, 1):(3, 1, 0)
```

### Coalesce

```julia
julia> layout = make_layout((2,(1,6)), (1,(6,2)))
(2, (1, 6)):(1, (6, 2))

julia> print(coalesce(layout))
12:1
```

### Composition
```julia
julia> make_layout(20,2) ∘ make_layout((4,5),(1,4))
(4, 5):(2, 8)
```

### Complement

### Product

### Division