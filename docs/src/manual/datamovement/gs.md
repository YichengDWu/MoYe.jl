# Global to Shared Memory Copy

This tutorial demonstrates how to copy data between global and shared memory using `MoYe.jl`. We will use the following configuration:

- Array size: 2048 x 2048
- Block size: 32 x 32
- Thread size: 32 x 8

## Copy Kernel

We start with a simple copy kernel:
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

    blocklayout = @Layout (32, 32) # 32x32 elements per block
    smemlayout = @Layout (32, 32)  # 32x32 elements in shared memory
    threadlayout = @Layout (32, 8) # 32x8 threads per block

    bM = size(blocklayout, 1)
    bN = size(blocklayout, 2)

    blocks = (cld(M, bM), cld(N, bN))
    threads = Int(size(threadlayout))

    @cuda blocks=blocks threads=threads copy_kernel(a, b, smemlayout, blocklayout, threadlayout)
    CUDA.synchronize()
    @test a == b
end

test_copy_async(2048, 2048)
```

### Code Explanation

The device function performs the following steps:

1.  **Allocate shared memory**: `MoYeSharedArray` allocates shared memory with a static layout.
2.  **Wrap arrays**: The destination and source arrays are wrapped with dynamic layouts.
3.  **Get block size**: Get the size of each block in the grid (`bM` and `bN`).
4.  **Create local tiles**: Create local tiles for the destination and source arrays using [`@tile`](@ref).
5.  **Partition tiles**: Partition the local tiles into thread tiles using [`@parallelize`](@ref).
6.  **Copy to shared memory**: Copy data from the source thread tile to the shared memory thread tile.
7.  **Synchronize threads**.
8.  **Copy to destination**: Copy data from the shared memory thread tile to the destination thread tile.

The host function tests the `copy_kernel` function:

1.  **Define dimensions**: Define the dimensions `M` and `N` for the source and destination arrays.
2.  **Create GPU arrays**: Create random GPU arrays `a` and `b`.
3.  **Define layouts**: Define the block and thread layouts using [`@Layout`](@ref).
4.  **Calculate grid size**: Calculate the number of blocks in the grid using `cld`.

Key points to note:

1.  [`@tile`](@ref) ensures that all blocks cover the entire array.
2.  Each block contains 32x32 elements, but we have 32x8 threads per block, so each thread processes 4 elements. The code `@parallelize blocktile_dest threadlayout threadIdx().x` returns the set of elements that the current thread is responsible for, which is an array of length 4.
3.  Once tiling is complete, we can perform computations as if we were working with a regular array:

```julia
for i in eachindex(threadtile_smem)
    threadtile_smem[i] = threadtile_src[i]
end
```

MoYe.jl handles the index bookkeeping implicitly, allowing you to focus on the computation.

### Using `copyto!`

You can also use [`copyto!`](@ref) for static `MoYeArray`s. This function automatically calls `cp.async` when copying from global to shared memory (requires `sm_80` or higher) and performs automatic vectorization when possible.

Here is the kernel using `copyto!`:
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

    copyto!(threadtile_smem, threadtile_src)
    cp_async_wait()
    copyto!(threadtile_dest, threadtile_smem)

    return nothing
end
```

## Padding Shared Memory

In the above code, the shared memory layout is the same as the block layout. However, it is often necessary to pad the shared array to avoid **bank conflicts**. This can be done by changing one line of code:
```julia
smemlayout = @Layout (32, 32) (1, 31)  # Pad one row
```
Note that the kernel will recompile for different static layout parameters.

## Transpose Kernel

Now, let's look at a transpose kernel:
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

    copyto!(threadtile_smem, threadtile_src)
    cp_async_wait()
    sync_threads()

    moye_smem′ = MoYe.transpose(moye_smem)
    threadtile_smem′ = @parallelize moye_smem′ threadlayout threadIdx().x

    copyto!(threadtile_dest, threadtile_smem′)
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
    threads = Int(size(threadlayout))

    @cuda blocks=blocks threads=threads transpose_kernel(a, b, smemlayout, blocklayout, threadlayout)
    CUDA.synchronize()
    @test a == transpose(b)
end

test_transpose(2048, 2048)
```

This is almost identical to the copy kernel, but we transpose the shared memory by transposing its layout:
```julia
    moye_smem′ = MoYe.transpose(moye_smem)
```
We then compute the new thread tiles. Note that each thread will now work on different elements, so we need to call `sync_threads()`.