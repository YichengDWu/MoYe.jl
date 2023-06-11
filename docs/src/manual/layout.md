# Layout 

Mathematically, a `Layout` represents a function that maps a logical coordinate to a 1-D index space that can be used to index into an array. It consists of a `shape` and a `stride`, wherein the `shape` determines the domain, and the `stride` establishes the mapping through an inner product. `shape` and `stride`  are both defined by (recursive) tuples of integers.

For example, we can construct a vector with stride 2 
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
print("Size: ", size(layout_2x4)) # the domain is (1,2,...,8)
print("Rank: ", rank(layout_2x4))
print("Depth: ", depth(layout_2x4))
print("Cosize: ", cosize(layout_2x4)) 
(layout_2x4) # this can be viewed as a row-major matrix
```

### Compile-time-ness of values

You can also use static integers:

```@repl layout
static_layout = @Layout (2, (2, 2)) (4, (1, 2))
typeof(static_layout)
sizeof(static_layout)

```

#### Different results from static Layout vs dynamic Layout

It is expected to get results that **appears** to be different when the layout 
is static or dynamic. For example,

```@repl layout
layout = @Layout (2, (1, 6)) (1, (6, 2)) 
print(coalesce(layout))
```

is different from

```@repl layout
layout = Layout((2, (1, 6)), (1, (6, 2))) 
print(coalesce(layout))
```
But they **are** mathematically equivalent. Static information allows us to simplify the
result as much as possible, whereas dynamic layouts result in dynamic checking hence type 
instability. 

## Coordinate space

The coordinate space of a `Layout` is determined by its `Shape`. This coordinate space can be viewed in three different ways:

 1. h-D coordinate space: Each element in this space possesses the exact hierarchical structure as defined by the Shape. Here `h` stands for "hierarchical".
 2. 1-D coordinate space: This can be visualized as the colexicographically flattening of the coordinate space into a one-dimensional space.
 3. R-D coordinate space: In this space, each element has the same rank as the Shape, but each mode (top-level axis) of the `Shape` is colexicographically flattened into a one-dimensional space. Here `R` stands for the rank of the layout.

```@repl layout
layout_2x4(2, (1, 2)) # h-D coordinate
layout_2x4(2, 3) # R-D coordinate
layout_2x4(6) # 1-D coordinate
```
## Layout Algebra

### Concatenation

A `layout` can be expressed as the concatenation of its sublayouts.

```@repl layout
layout_2x4[2] # get the second sublayout
tuple(layout_2x4...) # splatting a layout into sublayouts
make_layout(layout_2x4...) # concatenating sublayouts
for sublayout in layout_2x4 # iterating a layout
   @show sublayout
end
```



### Complement
Let's assume that we are dealing with a vector of 24 elements.
Our goal is to partition this vector into six tiles, each consisting of four elements, following a specific pattern 

```@exmaple complement
import Term: Panel # hide
panel = Panel(fit=true) # hide
red_panel = Panel(style="red",fit=true)  # hide
red_panel * panel * red_panel * panel * red_panel * panel * red_panel * panel * foldl(*, [panel for _ in 1:16])  # hide
```

This operation creates a new layout where we collect every second element until we have four elements, and then repeat this process for the rest of the vector.

The resulting layout would resemble:

```julia
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
print_layout(complement(Layout(4,2), 24))
```

The layout `Layout(4,2)` and it complement gives us the desired new layout.

```@repl layout
print_layout(make_layout(Layout(4, 2),complement(Layout(4, 2), 24)))
```

### Product

#### Logical product

```@repl layout
tile = @Layout((2,2), (1,2));
print_layout(tile)
matrix_of_tiles = @Layout((3,4), (4,1));
print_layout(matrix_of_tiles)
print_layout(logical_product(tile, matrix_of_tiles))
```

#### Blocked product

```@repl layout
print_layout(blocked_product(tile, matrix_of_tiles))
```

#### Raked product

```@repl layout
print_layout(raked_product(tile, matrix_of_tiles))
```

### Division

#### Logical division

```@repl layout
raked_prod = raked_product(tile, matrix_of_tiles);
subtile = (Layout(2,3), Layout(2,4));
print_layout(logical_divide(raked_prod, subtile))
```

#### Zipped division

```@repl layout
print_layout(zipped_divide(raked_prod, subtile))
```
