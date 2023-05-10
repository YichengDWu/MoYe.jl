# Broadcasting

For now only [`MoYeArray`](@ref) with static layouts are supported.

```@repl bc
using MoYe
a = MoYeArray{Float64}(undef, @Layout((3,2)))
fill!(a, 1.0)
a .* 3
a .+ a
```

Note that if `ndims(a)>ndims(b)`, the layout of `a` wins.

```@repl bc
b = MoYeArray{Float64}(undef, @Layout((3,), (2,))) # the stride is 2
a .- b # but the layout of b is ignored
```

There is limited support for broadcasting on device:
```julia
julia> function f()
           a = MoYeArray{Float64}(undef, @Layout((3,2)))
           fill!(a, one(eltype(a)))
           a .= a .* 2
           @cushow sum(a)
           b = CUDA.exp.(a)
           @cushow sum(b)
           return nothing
       end
f (generic function with 1 method)

julia> @cuda f()
sum(a) = 12.000000
sum(b) = 44.334337
CUDA.HostKernel{typeof(f), Tuple{}}(f, CuFunction(Ptr{CUDA.CUfunc_st} @0x0000026e00ca1af0, CuModule(Ptr{CUDA.CUmod_st} @0x0000026e15cfc900, CuContext(0x0000026da1fff8b0, instance e5a1871b578f5adb))), CUDA.KernelState(Ptr{Nothing} @0x0000000204e00000))
```