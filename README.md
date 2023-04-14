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

### Concatenation

A layout can be expressed as the concatenation of its sublayouts.

```julia
julia> layout_2x4[2] # get the second sublayout
(2, 2):(1, 2)

julia> tuple(layout_2x4...) # splatting a layout into sublayouts
(2:4, (2, 2):(1, 2))

julia> make_layout(layout_2x4...) # concatenating sublayouts
(2, (2, 2)):(4, (1, 2))

julia> for sublayout in layout_2x4 # iterating a layout
       @show sublayout
       end
sublayout = 2:4
sublayout = (2, 2):(1, 2)

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
julia> print_layout(tile)
(2, 2):(1, 2)
      1   2
    +---+---+
 1  | 1 | 3 |
    +---+---+
 2  | 2 | 4 |
    +---+---+

julia> matrix_of_tiles = make_layout((3,4), (4,1));
julia> print_layout(matrix_of_tiles)
(3, 4):(4, 1)
       1    2    3    4
    +----+----+----+----+
 1  |  1 |  2 |  3 |  4 |
    +----+----+----+----+
 2  |  5 |  6 |  7 |  8 |
    +----+----+----+----+
 3  |  9 | 10 | 11 | 12 |
    +----+----+----+----+


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

#### Blocked product
```julia
julia> print_layout(blocked_product(tile, matrix_of_tiles))
((2, 3), 8):((1, 16), 2)
       1    2    3    4    5    6    7    8
    +----+----+----+----+----+----+----+----+
 1  |  1 |  3 |  5 |  7 |  9 | 11 | 13 | 15 |
    +----+----+----+----+----+----+----+----+
 2  |  2 |  4 |  6 |  8 | 10 | 12 | 14 | 16 |
    +----+----+----+----+----+----+----+----+
 3  | 17 | 19 | 21 | 23 | 25 | 27 | 29 | 31 |
    +----+----+----+----+----+----+----+----+
 4  | 18 | 20 | 22 | 24 | 26 | 28 | 30 | 32 |
    +----+----+----+----+----+----+----+----+
 5  | 33 | 35 | 37 | 39 | 41 | 43 | 45 | 47 |
    +----+----+----+----+----+----+----+----+
 6  | 34 | 36 | 38 | 40 | 42 | 44 | 46 | 48 |
    +----+----+----+----+----+----+----+----+
```

#### Raked product
```julia
julia> print_layout(raked_product(tile, matrix_of_tiles))
((3, 2), (4, 2)):((16, 1), (4, 2))
       1    2    3    4    5    6    7    8
    +----+----+----+----+----+----+----+----+
 1  |  1 |  5 |  9 | 13 |  3 |  7 | 11 | 15 |
    +----+----+----+----+----+----+----+----+
 2  | 17 | 21 | 25 | 29 | 19 | 23 | 27 | 31 |
    +----+----+----+----+----+----+----+----+
 3  | 33 | 37 | 41 | 45 | 35 | 39 | 43 | 47 |
    +----+----+----+----+----+----+----+----+
 4  |  2 |  6 | 10 | 14 |  4 |  8 | 12 | 16 |
    +----+----+----+----+----+----+----+----+
 5  | 18 | 22 | 26 | 30 | 20 | 24 | 28 | 32 |
    +----+----+----+----+----+----+----+----+
 6  | 34 | 38 | 42 | 46 | 36 | 40 | 44 | 48 |
    +----+----+----+----+----+----+----+----+
```
### Division
