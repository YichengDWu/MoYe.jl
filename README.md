# CuTe

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://YichengDWu.github.io/CuTe.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://YichengDWu.github.io/CuTe.jl/dev/)
[![Build Status](https://github.com/YichengDWu/CuTe.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/YichengDWu/CuTe.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/YichengDWu/CuTe.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/YichengDWu/CuTe.jl)

Please also refer to NVIDIA's [CuTe](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/00_quickstart.md) for tutorials.

## Layout

Mathematically, a Layout represents a function that maps logical coordinates to physical index spaces. It consists of a Shape and a Stride, wherein the Shape determines the domain, and the Stride establishes the mapping through an inner product.

### Constructing a `Layout`

```julia
julia> layout_2x4 = make_layout((2,(2,2)),(4,(1,2)))
(2, (2, 2)):(4, (1, 2))

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

### Coordinate space

The coordinate space of a Layout is determined by its Shape. This coordinate space can be viewed in three distinct ways:

1. H-D coordinate space: Each element in this space possesses the exact hierarchical structure as defined by the Shape.
2. 1-D coordinate space: This can be visualized as the colexicographically flattening of the coordinate space into a one-dimensional space.
3. R-D coordinate space: In this space, each element has the same rank as the Shape, but each mode (axis) of the Shape is colexicographically flattened into a one-dimensional space.

```julia
julia> layout_2x4(2, (1,2)) # H-D coordinate
7

julia> layout_2x4(2,3) # R-D coordinate
7

julia> layout_2x4(6) # 1-D coordinate
7
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

julia> make_layout(20,2) ∘ make_layout((4,5),(5,1))
(4, 5):(10, 2)
```

### Complement

```julia
julia> complement(make_layout(4,1), 24)
6:4

julia> complement(make_layout(6,4), 24)
4:1
```
### Product

#### Logical product
```julia
julia> tile = make_layout((2,2), (1,2));

julia> matrix_of_tiles = make_layout((3,4), (4,1));

julia> print_layout(logical_product(tile, matrix_of_tiles));
((2, 2), (3, 4)):((1, 2), (16, 4))
       1    2    3    4    5    6    7    8    9   10   11   12
    +----+----+----+----+----+----+----+----+----+----+----+----+
 1  |  1 | 17 | 33 |  5 | 21 | 37 |  9 | 25 | 41 | 13 | 29 | 45 |
    +----+----+----+----+----+----+----+----+----+----+----+----+
 2  |  2 | 18 | 34 |  6 | 22 | 38 | 10 | 26 | 42 | 14 | 30 | 46 |
    +----+----+----+----+----+----+----+----+----+----+----+----+
 3  |  3 | 19 | 35 |  7 | 23 | 39 | 11 | 27 | 43 | 15 | 31 | 47 |
    +----+----+----+----+----+----+----+----+----+----+----+----+
 4  |  4 | 20 | 36 |  8 | 24 | 40 | 12 | 28 | 44 | 16 | 32 | 48 |
    +----+----+----+----+----+----+----+----+----+----+----+----+

```

### Division
