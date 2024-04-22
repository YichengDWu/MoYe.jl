# MatMul
![matmuil](../assets/matmul.png)

In this tutorial, we explore matrix multiplication using MoYe.jl , specifically computing the product $C = A * B^\top$. Here, matrix $A$ has dimensions $(M, K)$, matrix $B$ has dimensions $(K, N)$, and the resulting matrix $C$ will have dimensions $(M, N)$.

First, we divide the task among each block. We use a tile of size (bM, bN) to partition C, with each block responsible for computing one tile. The tile's index is determined by (blockIdx().x, blockIdx().y).

Computing a tile requires all values from A of shape (dM, K) and B of shape (dN, K). To reduce global memory access (since A, B, and C are stored in global memory), we further partition A and B along the K dimension, sequentially loading elements of sizes (dM, dK) and (dN, dK) into shared memory, then performing the matrix multiplication and accumulating the results into the tile of C.

The partition of the global memory corresponds to the following three lines of code:
```julia
gC = @tile C (bM, bN) (blockIdx().x, blockIdx().y) # (bM, bN)
gA = @tile A (bM, bK) (blockIdx().x, :)            # (bM, bK, K/bK)
gB = @tile B (bN, bK) (blockIdx().y, :)            # (bN, bK, K/bK)
```
For the specific partition syntax, please refer to [`@tile`](@ref). Here, `gA` represents `A` in shared memory. Next, we use a for loop to index-slice the last dimension of gA and gB (denoted as `k`), loading them into shared memory. The code for this step is:

```julia
sA = MoYeSharedArray(eltype(gA), sA_layout) # (bM, bK)
sB = MoYeSharedArray(eltype(gB), sB_layout) # (bN, bK)
```

`MoYeSharedArray` automatically allocates shared memory of size `cosize(sA_layout) + cosize(sB_layout)` and returns a `MoYeArray`. We will explain how to define the layouts for sA and sB later; for now, it's only necessary to know that they are predefined at compile time.

We then need to define how thread groups collectively copy from global to shared memory. There are many ways to organize threads, which will be discussed later, such as:
```julia
tA = @Layout (32, 8)
tB = @Layout (32, 8)
```

This implies that there are 32x8 threads arranged in a column-major format. Next, we use them to partition the arrays:
```julia
tAgA = @parallelize tA threadIdx().x
tBgB = @parallelize tB threadIdx().x

tAsA = @parallelize sA threadIdx().x
tBsB = @parallelize sB threadIdx().x
```

For the specific syntax, please refer to [`@parallelize`](@ref). After the partition, copying is simply:
```julia
copyto!(tAsA, view(tAgA, :, :, k))
copyto!(tBsB, view(tBgB, :, :, k))
```

After copying, we proceed to the actual matrix-multiply-accumulate (mma) computation. Similarly, we need to define a layout for the thread group for this purpose:
```julia
tC = @Layout (16, 16)
```

Then we use it to partition gC:
```julia
tCgC = @parallelize gC tC threadIdx().x 
tCrC = similar(tCgC)
```

To reduce memory access to C, we also create an array `tCrC` stored in registers, which serves as the accumulator in the mma computation. After the computation, the contents are copied back into `tCgC`.

A and B are slightly different because computing an element in C requires an entire row from A and an entire column from B, which is reflected in the following code:

```julia
tCsA = @parallelize sA tC threadIdx().x (1, :) 
tCsB = @parallelize sB tC threadIdx().x (:, 1)
```

Congratulations, you have now completed all the partitions, and finally, we can compute the matrix multiplication, just as we would on a CPU:

```julia
for k in axes(tCsA, 2)
    for m in axes(tCsA, 1)
        for n in axes(tCsB, 1)
            @inbounds tCrC[m, n] += tCsA[m, k] * tCsB[n, k]
        end
    end
end
```
You can also call [`gemm!`] to perform the same operation:
```julia
gemm!(tCsA, tCsB, tCrC)
```

The complete kernel code is as follows:
```julia
function matmul_kernel(A, sA_layout, tA,
                       B, sB_layout, tB,
                       C, tC)
    M = size(A, 1)
    N = size(B, 1)
    K = size(A, 2)

    bM = size(sA_layout, 1)
    bN = size(sB_layout, 1)
    bK = size(sB_layout, 2)

    sA = MoYeSharedArray(eltype(A), sA_layout)
    sB = MoYeSharedArray(eltype(B), sB_layout)

    mA = MoYeArray(A, (M, K))
    mB = MoYeArray(B, (N, K))
    mC = MoYeArray(C, (M, N))

    gA = @tile mA (bM, bK) (blockIdx().x, :)
    gB = @tile mB (bN, bK) (blockIdx().y, :)
    gC = @tile mC (bM, bN) (blockIdx().x, blockIdx().y)

    # copy partition
    tAgA = @parallelize gA tA threadIdx().x 
    tBgB = @parallelize gB tB threadIdx().x
    tAsA = @parallelize sA tA threadIdx().x
    tBsB = @parallelize sB tB threadIdx().x

    # mma partition
    tCsA = @parallelize sA tC threadIdx().x (1, :) 
    tCsB = @parallelize sB tC threadIdx().x (:, 1)
    tCgC = @parallelize gC tC threadIdx().x 

    # accumulator
    tCrC = similar(tCgC)
    zeros!(tCrC)

    for k in axes(tAgA, 3)
        copyto!(tAsA, view(tAgA, :, :, k))
        copyto!(tBsB, view(tBgB, :, :, k))
        
        cp_async_wait()
        sync_threads()

        @gc_preserve gemm!(tCsA, tCsB, tCrC)
        sync_threads()
    end


    copyto!(tCgC, tCrC)
    return nothing
end

```

We still missed a few points, such as:

1. How to design `sA_layout` and `sB_layout`?

For shared memory, we no longer need to consider column-major or row-major but simply need to avoid bank conflicts. This can be simply achieved by padding one column.

```julia
sA_layout = @Layout (bM, bK) (_1, bM + _1)
sB_layout = @Layout (bN, bK) (_1, bN + _1)
```

2. How to design `tC`?

The design of `tC` is quite flexible; it only needs to satisfy that the shape of `tC` evenly divides `(bM, bN)`.

3. How to design `tA` and `tB`?

You generally want every 32 threads to access contiguous elements in A and B, so the specific design depends on the memory layout of A and B. This technique is known as memory coalescing.

The `matmul` function looks like this:

```julia
function matmul(A, B, C)
    bM = _128
    bN = _128
    bK = _8
    
    sA_layout = make_layout((bM, bK))
    sB_layout = make_layout((bN, bK))

    tA = @Layout (32, 8)
    tB = @Layout (32, 8)
    tC = @Layout (16, 16)

    threads = Int(size(tC))
    blocks = (cld(size(A, 1), bM), cld(size(B, 1), bN))

    @cuda threads=threads blocks=blocks matmul_kernel(A, sA_layout, tA,
                                                      B, sB_layout, tB,
                                                      C, tC)
end
```

This concludes the guide to implementing matrix multiplication with MoYe.jl, focusing on efficient memory management and