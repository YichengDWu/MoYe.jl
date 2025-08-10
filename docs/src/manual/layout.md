# Layout

A `Layout` defines how multidimensional data is stored in one-dimensional memory. It maps a logical coordinate to a linear index using a `shape` and a `stride`. The `shape` defines the dimensions of the array, while the `stride` determines the memory offset for each dimension.

For example, let's create a vector with a stride of 2:
```@repl layout
using MoYe
struct StrideVector
   data
   layout
end

Base.getindex(x::StrideVector, i) = x.data[x.layout(i)]
a = StrideVector(collect(1:8), Layout(4, 2))
@show a[1] a[2] a[3] a[4];
```

## Fundamentals

```@repl layout
using MoYe
layout_2x4 = Layout((2, (2, 2)), (4, (1, 2)))
print("Shape: ", shape(layout_2x4))
print("Stride: ", stride(layout_2x4))
print("Size: ", size(layout_2x4)) # The domain is 1:8
print("Rank: ", rank(layout_2x4))
print("Depth: ", depth(layout_2x4))
print("Cosize: ", cosize(layout_2x4)) 
layout_2x4 # This can be viewed as a row-major matrix
```

### Compile-Time vs. Dynamic Layouts

You can also use static integers for compile-time layouts:

```@repl layout
static_layout = @Layout (2, (2, 2)) (4, (1, 2))
typeof(static_layout)
sizeof(static_layout)

```

Static and dynamic layouts can produce different-looking but mathematically equivalent results. For example:

```@repl layout
layout = @Layout (2, (1, 6)) (1, (6, 2)) 
print(coalesce(layout))
```

is different from:

```@repl layout
layout = Layout((2, (1, 6)), (1, (6, 2))) 
print(coalesce(layout))
```
Static layouts allow for more aggressive compile-time simplification, while dynamic layouts may lead to type instability due to runtime checks.

## Coordinate Spaces

A `Layout`'s coordinate space is determined by its `Shape` and can be viewed in three ways:

1.  **h-D (Hierarchical) Coordinate Space**: Each element has the same hierarchical structure as the `Shape`.
2.  **1-D Coordinate Space**: The colexicographically flattened, one-dimensional representation of the coordinate space.
3.  **R-D Coordinate Space**: Each element has the same rank as the `Shape`, but each top-level axis is colexicographically flattened into a one-dimensional space. `R` is the rank of the layout.

```@repl layout
layout_2x4(2, (1, 2)) # h-D coordinate
layout_2x4(2, 3)      # R-D coordinate
layout_2x4(6)         # 1-D coordinate
```

## Layout Algebra

### Concatenation

A `Layout` can be expressed as the concatenation of its sub-layouts.

```@repl layout
layout_2x4[2] # Get the second sub-layout
tuple(layout_2x4...) # Splat a layout into its sub-layouts
make_layout(layout_2x4...) # Concatenate sub-layouts
for sublayout in layout_2x4 # Iterate over sub-layouts
   @show sublayout
end
```

### Complement

Let's partition a vector of 24 elements into six tiles of four elements each, gathering every fourth element at even indices.

This operation creates a new layout where we collect every second element until we have four, then repeat this for the rest of the vector.

The resulting layout would resemble:

```
       1    2    3    4    5    6
    +----+----+----+----+----+----+
 1  |  1 |  2 |  9 | 10 | 17 | 18 |
    +----+----+----+----+----+----+
 2  |  3 |  4 | 11 | 12 | 19 | 20 |
    +----+----+----+----+----+----+
 3  |  5 |  6 | 13 | 14 | 21 | 22 |
    +----+----+----+----+----+----+
 4  |  7 |  8 | 15 | 16 | 23 | 24 |
    +----+----+----+----+----+----+
```

`complement` computes the first row of this new layout.

```@repl layout
print_layout(complement(@Layout(4,2), 24))
```

The `Layout(4,2)` and its complement give us the desired new layout:

```@repl layout
print_layout(make_layout(@Layout(4, 2),complement(@Layout(4, 2), 24)))
```

### Product

#### Logical Product

```@repl layout
tile = @Layout((2,2), (1,2));
print_layout(tile)
matrix_of_tiles = @Layout((3,4), (4,1));
print_layout(matrix_of_tiles)
print_layout(logical_product(tile, matrix_of_tiles))
```

#### Blocked Product

```@repl layout
print_layout(blocked_product(tile, matrix_of_tiles))
```

#### Raked Product

```@repl layout
print_layout(raked_product(tile, matrix_of_tiles))
```

### Division

#### Logical Division

```@repl layout
raked_prod = raked_product(tile, matrix_of_tiles);
subtile = (@Layout(2,3), @Layout(2,4));
print_layout(logical_divide(raked_prod, subtile))
```

#### Zipped Division

```@repl layout
print_layout(zipped_divide(raked_prod, subtile))
```