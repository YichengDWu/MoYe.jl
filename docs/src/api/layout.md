# Layout 

```@meta
CurrentModule = MoYe
```
## Index

```@index
Pages = ["layout.md"]
```

## Constructors
```@docs
Layout
@Layout
make_layout
```

## Fundamentals

```@docs
size(::Layout)
rank(::Layout)
depth(::Layout)
cosize(::Layout)
getindex(layout::Layout, Is::IntType...)
```
## Compact Layout

```@docs
GenColMajor
GenRowMajor
```
## Algebra

### Concatenation 
```@docs
cat(::Layout...)
make_layout(::Layout...)
```
### Composition
```@docs
composition
```
### Complement
```@docs
complement
```
### Inverse
```@docs
left_inverse
right_inverse
```
### Product
```@docs
logical_product
blocked_product
raked_product
```
### Division 
```@docs
logical_divide
zipped_divide
tiled_divide
```

### Miscellaneous
```@docs
coalesce
flatten(layout::Layout)
```