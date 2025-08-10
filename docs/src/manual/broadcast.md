# Broadcasting

Broadcasting is a powerful feature that allows you to perform element-wise operations on arrays of different shapes and sizes. In MoYe.jl, broadcasting is defined for `MoYeArray`s with static sizes.

## In-Place vs. Out-of-Place Broadcasting

-   **In-place broadcasting** (`.=`, `.+=`, etc.) modifies the original array and preserves its layout.
-   **Out-of-place broadcasting** (`.`, `+`, etc.) returns a new array with a compact layout, the same shape, and the same stride order.

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

## Broadcasting on the GPU

In-place broadcasting on the GPU works seamlessly:

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