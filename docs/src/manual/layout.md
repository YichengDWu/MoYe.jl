# Layout 

Mathematically, a `Layout` represents a function that maps logical coordinates to physical 1-D index spaces. It consists of a `Shape` and a `Stride`, wherein the `Shape` determines the domain, and the `Stride` establishes the mapping through an inner product.

## Constructing a `Layout`

```@repl layout
using MoYe
layout_2x4 = make_layout((2, (2, 2)), (4, (1, 2)))
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


## Concatenation

A `layout` can be expressed as the concatenation of its sublayouts.

```@repl layout
layout_2x4[2] # get the second sublayout
tuple(layout_2x4...) # splatting a layout into sublayouts
make_layout(layout_2x4...) # concatenating sublayouts
for sublayout in layout_2x4 # iterating a layout
   @show sublayout
end
```


## Flatten

```@repl layout
layout = make_layout(((4, 3), 1), ((3, 1), 0))
print(flatten(layout))
```

### Coalesce

```@repl layout
layout = @Layout (2, (1, 6)) (1, (6, 2)) # layout has to be static
print(coalesce(layout))
```

## Composition

Layouts are functions and thus can possibly be composed.
```@repl layout
make_layout(20, 2) ∘ make_layout((4, 5), (1, 4)) 
make_layout(20, 2) ∘ make_layout((4, 5), (5, 1))
```

## Complement

```@repl layout
complement(@Layout(4, 1), static(24))
complement(@Layout(6, 4), static(24))
```

## Product

### Logical product

```@repl layout
tile = @Layout((2,2), (1,2));
print_layout(tile)
matrix_of_tiles = @Layout((3,4), (4,1));
print_layout(matrix_of_tiles)
print_layout(logical_product(tile, matrix_of_tiles))
```

### Blocked product

```@repl layout
print_layout(blocked_product(tile, matrix_of_tiles))
```

### Raked product

```@repl layout
print_layout(raked_product(tile, matrix_of_tiles))
```

## Division

### Logical division

```@repl layout
raked_prod = raked_product(tile, matrix_of_tiles);
subtile = (Layout(2,3), Layout(2,4));
print_layout(logical_divide(raked_prod, subtile))
```

### Zipped division

```@repl layout
print_layout(zipped_divide(raked_prod, subtile))
```
