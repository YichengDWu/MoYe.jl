# Broadcasting

# For now only [`MoYeArray`](@ref) with static layouts are supported.


```@repl bc
using MoYe
a = MoYeArray{Float64}(undef, @Layout((3,2)))
fill!(a, 1.0)
a .* 3
a .+ a
```

# Note that if `ndims(a)>ndims(b)`, the layout of `a` wins.

```@repl bc
b = MoYeArray{Float64}(undef, @Layout((3,), (2,))) # the stride is 2
a .- b # but the layout of b is ignored
```
