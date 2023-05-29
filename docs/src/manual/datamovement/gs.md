# Matrix Transpose Tutorial

This tutorial illustrates the process copying data between global memory and shared memory using `MoYe`. 

In this tutorial, we will use the following configuration:

- Array size: 2048 x 2048
- Block size: 32 x 32
- Thread size: 32 x 8

## Copy Kernel
We start with a copy kernel.
```julia
using MoYe, Test, CUDA

function copy_kernel(dest, src, smemlayout, blocklayout, threadlayout)
    moye_smem = MoYeSharedArray(eltype(dest), smemlayout) 

    moye_dest = MoYeArray(dest)
    moye_src = MoYeArray(src)

    bM = size(blocklayout, 1)
    bN = size(blocklayout, 2)

    blocktile_dest = @tile moye_dest (bM, bN) (blockIdx().x, blockIdx().y)
    blocktile_src  = @tile moye_src  (bM, bN) (blockIdx().x, blockIdx().y)

    threadtile_dest = @parallelize blocktile_dest threadlayout threadIdx().x
    threadtile_src  = @parallelize blocktile_src  threadlayout threadIdx().x
    threadtile_smem = @parallelize moye_smem      threadlayout threadIdx().x

    for i in eachindex(threadtile_smem)
        threadtile_smem[i] = threadtile_src[i]
    end
    
    for i in eachindex(threadtile_dest)
        threadtile_dest[i] = threadtile_smem[i]
    end
    return nothing
end

function test_copy_async(M, N)
    a = CUDA.rand(Float32, M, N)
    b = CUDA.rand(Float32, M, N)

    blocklayout = @Layout (32, 32) # 32 * 32 elements in a block
    smemlayout = @Layout (32, 32)  # 32 * 32 elements in shared memory
    threadlayout = @Layout (32, 8) # 32 * 8 threads in a block

    bM = size(blocklayout, 1)
    bN = size(blocklayout, 2)

    blocks = (cld(M, bM), cld(N, bN))
    threads = MoYe.dynamic(size(threadlayout))

    @cuda blocks=blocks threads=threads copy_kernel(a, b, smemlayout, blocklayout, threadlayout)
    CUDA.synchronize()
    @test a == b
end

test_copy_async(2048, 2048)
```
## Code Explanation

The device function follows these steps:

1. Allocate shared memory using `MoYeSharedArray` with a static layout.
2. Wrap the destination and source arrays with dynamic layouts.
3. Get the size of each block in the grid (bM and bN).
4. Create local tiles for the destination and source arrays using [`@tile`](@ref).
5. Partition the local tiles into thread tiles using [`@parallelize`](@ref).
6. Copy data from the source thread tile to the shared memory thread tile.
7. Synchronize threads.
8. Copy data back from the shared memory thread tile to the destination thread tile.

The host function tests the copy_kernel function with the following steps:

1. Define the dimensions M and N for the source and destination arrays.
2. Create random GPU arrays a and b with the specified dimensions using CUDA.rand.
3. Define the block and thread layouts using [`@Layout`](@ref) for creating **static** layouts.
4. Calculate the number of blocks in the grid using `cld`. Here we assume the divisibility.

A few things to notice here:

1. [`@tile`](@ref) means that all of our blocks cover the entire array.
2. Each block contains 32 x 32 elements of the original array, but we have 32 x 8 threads per block, which means that each thread processes 4 elements. The code
```julia
@parallelize blocktile_dest threadlayout threadIdx().x
```
returns the set of elements that the thread corresponding to threadIdx().x is processing, which in this case is an array of length 4.

3. Once we have completed all the tiling, we just perform computations as if we were dealing with a regular array:

```julia
for i in eachindex(threadtile_smem)
    threadtile_smem[i] = threadtile_src[i]
end
```
You need not concern yourself with index bookkeeping, it is implicitly handled by the layout; instead, concentrate on the computation aspect, as it is a fundamental objective of MoYe.jl.

Additionally, you can use the [`cucopyto!`](@ref) function, which is similar to copyto!, but with two key differences: copying from global memory to shared memory automatically calls `cp.async` (Requires `sm_80` or higher), and automatic vectorization when possible.

Here is how it would look like using `cucopyto!`.
```julia
function copy_kernel(dest, src, smemlayout, blocklayout, threadlayout)
    moye_smem = MoYeSharedArray(eltype(dest), smemlayout) 

    moye_dest = MoYeArray(dest)
    moye_src = MoYeArray(src)

    bM = size(blocklayout, 1)
    bN = size(blocklayout, 2)

    blocktile_dest = @tile moye_dest (bM, bN) (blockIdx().x, blockIdx().y)
    blocktile_src  = @tile moye_src  (bM, bN) (blockIdx().x, blockIdx().y)

    threadtile_dest = @parallelize blocktile_dest threadlayout threadIdx().x
    threadtile_src  = @parallelize blocktile_src  threadlayout threadIdx().x
    threadtile_smem = @parallelize moye_smem      threadlayout threadIdx().x

    cucopyto!(threadtile_smem, threadtile_src)
    cp_async_wait()
    cucopyto!(threadtile_dest, threadtile_smem)

    return nothing
end
```

## Padding Shared Memory

Note that in the above code, the layout of the shared memory is the same as the block layout. However, we often need to pad the shared array to avoid bank conflicts. We just need to change one line of code:
```julia
smemlayout = @Layout (32, 32) (1, 31)  # pad one row
```
Also note that our kernel will recompile for different static layout parameters.

## Transpose kernel

Now we turn to the transpose kernel.
```julia
function transpose_kernel(dest, src, smemlayout, blocklayout, threadlayout)
    moye_smem = MoYeSharedArray(eltype(dest), smemlayout) 

    moye_src = MoYeArray(src)
    moye_dest = MoYeArray(dest)

    bM = size(blocklayout, 1)
    bN = size(blocklayout, 2)

    blocktile_src  = @tile moye_src  (bM, bN) (blockIdx().x, blockIdx().y)
    blocktile_dest = @tile moye_dest (bN, bM) (blockIdx().y, blockIdx().x)

    threadtile_dest = @parallelize blocktile_dest threadlayout threadIdx().x
    threadtile_src  = @parallelize blocktile_src  threadlayout threadIdx().x
    threadtile_smem = @parallelize moye_smem      threadlayout threadIdx().x

    cucopyto!(threadtile_smem, threadtile_src)
    cp_async_wait()
    sync_threads()

    moye_smem′ = MoYe.transpose(moye_smem)
    threadtile_smem′ = @parallelize moye_smem′ threadlayout threadIdx().x

    cucopyto!(threadtile_dest, threadtile_smem′)
    return nothing
end


function test_transpose(M, N)
    a = CUDA.rand(Float32, M, N)
    b = CUDA.rand(Float32, N, M)

    blocklayout = @Layout (32, 32)
    smemlayout = @Layout (32, 32) (1, 33)
    threadlayout = @Layout (32, 8)

    bM = size(blocklayout, 1)
    bN = size(blocklayout, 2)

    blocks = (cld(M, bM), cld(N, bN))
    threads = MoYe.dynamic(size(threadlayout))

    @cuda blocks=blocks threads=threads transpose_kernel(a, b, smemlayout, blocklayout, threadlayout)
    CUDA.synchronize()
    @test a == transpose(b)
end

test_transpose(2048, 2048)
```

It is almost identical to the copy kernel， but we would need to transpose the shared memory by simply transposing its layout
```julia
    moye_smem′ = MoYe.transpose(moye_smem)
```
and then compute the new thread tiles. Note that each thread would work on different elements now so we need to call `sync_threads()`.
