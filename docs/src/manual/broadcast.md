# Broadcasting

Broadcasting is only defined for [`MoYeArray`](@ref)s with static sizes. 

In-place broadcasting preserves the original layout.

Out-of-place broadcasting always returns an owning array of a compact layout with
the same shape and the stride ordered the same.

```@repl bc
using MoYe
a = MoYeArray{Float64}(undef, @Layout((3,2), (2,1)))
fill!(a, 1.0); 
a .* 3
a .+ a
```

```@repl bc
b = MoYeArray{Float64}(undef, @Layout((3,), (2,))) |> zeros!; # Create a vector
a .- b 
```
## On GPU
(In-place) broadcasting on device should just work:

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
