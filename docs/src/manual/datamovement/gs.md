# Copy Kernel Tutorial

This tutorial illustrates the process copying data between global memory and shared memory using `MoYe`. 

The copy kernel first asynchronously copies data from the global memory to the shared memory and subsequently validates the correctness of the operation by copying the data back from the shared memory to the global memory.

In this tutorial, we will use the following configuration:

- Array size: 256 x 32 (M x N)
- Block size: 128 x 16
- Thread size: 32 x 8

**Note**: Requires `sm_80` or higher.

```julia
using MoYe, Test, CUDA

function copy_kernel(M, N, dest, src, smemlayout, blocklayout, threadlayout)
    smem = MoYe.SharedMemory(eltype(dest), cosize(smemlayout))
    moye_smem = MoYeArray(smem, smemlayout)

    moye_dest = MoYeArray(pointer(dest), Layout((M, N), (static(1), M))) # bug: cannot use make_layout((M, N))
    moye_src = MoYeArray(pointer(src), Layout((M, N), (static(1), M)))

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
    sync_threads()
    return nothing
end

function test_copy_async(M, N)
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

test_copy_async(256, 32)
```
## Code Explanation

The device function follows these steps:

1. Allocate shared memory using `MoYe.SharedMemory`.
2. Wrap the shared memory with [`MoYeArray`](@ref) with a static layout and destination, and source arrays with dynamic layouts.
3. Compute the size of each block in the grid (bM and bN).
4. Create local tiles for the destination and source arrays using [`@tile`](@ref).
5. Partition the local tiles into thread tiles using [`@parallelize`](@ref).
6. Asynchronously copy data from the source thread tile to the shared memory thread tile using [`cucopyto!`](@ref).
7. Synchronize threads using [`cp_async_wait`](@ref).
8. Copy data back from the shared memory thread tile to the destination thread tile with `cucopyto!` again, but under the hood it is using the universal copy method.
9. Synchronize threads again using `sync_threads`.

The host function tests the copy_kernel function with the following steps:

1. Define the dimensions M and N for the source and destination arrays.
2. Create random GPU arrays a and b with the specified dimensions using CUDA.rand.
3. Define the block and thread layouts using [`@Layout`](@ref) for creating **static** layouts.
4. Calculate the number of blocks in the grid using `cld`. Here we assume the divisibility.


## Padding Shared Memory

Note that in the above code, the layout of the shared memory is the same as the block layout. However, we often need to pad the shared array to avoid bank conflicts. We just need to change one line of code:
```julia
smemlayout = @Layout (128, 16) (1, 130)  # pad 2 rows
```
Please note that our kernel will recompile for different static layout parameters.

## Transpose kernel

The following code transposes a matrix. To be more precise, it calculates a column-major transposed matrix.
```julia
using MoYe, Test, CUDA

function transpose_kernel(M, N, dest, src,
                          blocklayout, smemlayout,
                          threadlayout_src,threadlayout_dest)
    smem = MoYe.SharedMemory(eltype(dest), cosize(smemlayout))
    moye_smem = MoYeArray(smem, smemlayout)

    moye_src = MoYeArray(pointer(src), Layout((M, N), (static(1), M)))
    moye_dest = MoYeArray(pointer(dest), Layout((N, M), (static(1), N)))

    bM = size(blocklayout, 1)
    bN = size(blocklayout, 2)

    blocktile_src  = @tile moye_src  (bM, bN) (blockIdx().x, blockIdx().y)

    threadtile_src  = @parallelize blocktile_src  threadlayout_src threadIdx().x
    threadtile_smem = @parallelize moye_smem      threadlayout_src threadIdx().x

    cucopyto!(threadtile_smem, threadtile_src)
    cp_async_wait()

    # transpose smem
    moye_smem′ = MoYeArray(smem, transpose(smemlayout))

    blocktile_dest = @tile moye_dest (bN, bM) (blockIdx().y, blockIdx().x)

    threadtile_dest  = @parallelize blocktile_dest threadlayout_dest threadIdx().x
    threadtile_smem′ = @parallelize moye_smem′     threadlayout_dest threadIdx().x

    cucopyto!(threadtile_dest, threadtile_smem′)
    sync_threads()
    return nothing
end

function test_transpose(M, N)
    M = 256
    N = 32

    src = CUDA.rand(Float32, M, N)
    dest = CUDA.rand(Float32, N, M)

    blocklayout = @Layout (128, 16)
    smemlayout = @Layout (128, 16) (1, 130) 
    threadlayout_src = @Layout (32, 8)
    threadlayout_dest = @Layout (16, 16)

    bM = size(blocklayout, 1)
    bN = size(blocklayout, 2)

    blocks = (cld(M, bM), cld(N, bN))
    threads = MoYe.dynamic(size(threadlayout_src))

    @cuda blocks=blocks threads=threads transpose_kernel(M, N, dest, src, blocklayout, smemlayout,
                                                         threadlayout_src, threadlayout_dest)
    @test dest == transpose(src)
end

test_transpose(256, 32)
```

Now let us explain. 

First, we transfer the contents of global memory to shared memory, a process that closely resembles the previously implemented copy kernel function. Then, we perform a lazy transposition on the shared memory, which entails merely altering its layout to produce a row-major matrix.

```julia
moye_smem′ = MoYeArray(smem, transpose(smemlayout))
```

Following this, we copy the shared memory to global memory as usual. Note that at this point we swap `bM, bN` and `blockIdx().x, blockIdx().y`. Intriguingly, after this swap, our block size becomes 16 * 128, while the original thread layout remains 32*8, leading to a mismatch in their row numbers. This is where the power of layout comes into play: we can conveniently reconfigure the threads' layout multiple times within the kernel function to serve different purposes.
`threadlayout_src` `threadlayout_dest`