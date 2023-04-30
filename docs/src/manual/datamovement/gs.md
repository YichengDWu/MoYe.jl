# Copy Kernel Tutorial

This tutorial illustrates the process copying data between global memory and shared memory using `MoYe`. 

The copy kernel first asynchronously copies data from the global memory to the shared memory and subsequently validates the correctness of the operation by copying the data back from the shared memory to the global memory.

In this tutorial, we will use the following configuration:

- Array size: 256 x 32 (M x N)
- Block size: 128 x 16
- Thread size: 32 x 8


## Code Explanation

The device function follows these steps:

1. Allocate shared memory using `MoYe.SharedMemory`.
2. Wrap the shared memory with [`MoYeArray`](@ref) with a static layout and destination, and source arrays with dynamic layouts.
3. Compute the size of each block in the grid (bM and bN).
4. Create local tiles for the destination and source arrays using [`local_tile`](@ref).
5. Partition the local tiles into thread tiles using [`local_partition`](@ref).
6. Asynchronously copy data from the source thread tile to the shared memory thread tile using [`cucopyto!`](@ref).
7. Synchronize threads using `sync_threads`.
8. Copy data back from the shared memory thread tile to the destination thread tile with `cucopyto!` again, but under the hood it is using the universal copy method.
9. Synchronize threads again using `sync_threads`.

The host function tests the copy_kernel function with the following steps:

1. Define the dimensions M and N for the source and destination arrays.
2. Create random GPU arrays a and b with the specified dimensions using CUDA.rand.
3. Define the block and thread layouts using [`@Layout`] for creating **static** layouts.
4. Calculate the number of blocks in the grid using `cld`. Here we assume the divisibility.

```julia
using MoYe, Test, CUDA

function copy_kernel(M, N, dest, src, smemlayout, blocklayout, threadlayout)
    smem = MoYe.SharedMemory(eltype(dest), cosize(smemlayout))
    moye_smem = MoYeArray(smem, smemlayout)

    moye_dest = MoYeArray(pointer(dest), Layout((M, N), (static(1), M))) # bug: cannot use make_layout((M, N))
    moye_src = MoYeArray(pointer(src), Layout((M, N), (static(1), M)))

    bM = size(blocklayout, 1)
    bN = size(blocklayout, 2)

    blocktile_dest = @tile moye_dest (bM, bN) (Int(blockIdx().x), Int(blockIdx().y))
    blocktile_src  = @tile moye_src (bM, bN) (Int(blockIdx().x), Int(blockIdx().y))

    threadtile_dest = @parallelize blocktile_dest threadlayout Int(threadIdx().x)
    threadtile_src  = @parallelize blocktile_src  threadlayout Int(threadIdx().x)
    threadtile_smem = @parallelize moye_smem      threadlayout Int(threadIdx().x)

    cucopyto!(threadtile_smem, threadtile_src) 
    sync_threads()
    cucopyto!(threadtile_dest, threadtile_smem)
    sync_threads()
    return nothing
end

function test_copy_async()
    M = 256
    N = 32

    a = CUDA.rand(Float32, M, N)
    b = CUDA.rand(Float32, M, N)

    blocklayout = @Layout (128, 16)
    smemlayout = blocklayout
    threadlayout = @Layout (32, 8)

    bM = size(blocklayout, 1)
    bN = size(blocklayout, 2)

    blocks = (cld(M, bM), cld(N, bN))
    threads = MoYe.Static.dynamic(size(threadlayout))

    @cuda blocks=blocks threads=threads copy_kernel(M, N, a, b, smemlayout, blocklayout, threadlayout)
    @test a == b
end

test_copy_async()
```

## Padding Shared Memory

Note that in the above code, the layout of the shared memory is the same as the block layout. However, we often need to pad the shared array by one row to avoid bank conflicts. We just need to add one line of code:
```julia
smemlayout = @Layout (128, 16) (1,129)
```

Note that the stride is now 129, not 128. The rest of the code is basically identical.
