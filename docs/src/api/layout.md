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
cosize(::Layout)
```
## Compact Layout

```@docs
GenColMajor
GenRowMajor
```
## Algebra

### Concatenation 
```@docs
 cat(::Layouts...)
```
### Composition
```@docs
compose
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