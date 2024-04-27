## Memcpy Async

With the NVIDIA Ampere architecture, you can asynchronously copy data between GPU global memory and shared memory and not tie up threads to shepherd data movement.

To utilize this feature, we simply change the `TiledCopy` to the following 
```julia
copy_A = make_tiled_copy(CopyAtom{CPOP_ASYNC_CACHEALWAYS{TA}, TA}(),
                                @Layout((32, 8)),
                                @Layout((4, 1)))
copy_B = make_tiled_copy(CopyAtom{CPOP_ASYNC_CACHEALWAYS{TB}, TB}(),
                                    @Layout((32, 8)),
                                    @Layout((4, 1)))
```

The updated kernel function.

```julia
function matmul_kernel(A, sA_layout, copy_A,
                       B, sB_layout, copy_B,
                       C, mma_C)
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
    thr_copy_a = get_slice(copy_A, threadIdx().x)      
    tAgA = partition_S(thr_copy_a, gA)                 # (CPY, CPY_M, CPY_K, k)
    tAsA = partition_D(thr_copy_a, sA)                 # (CPY, CPY_M, CPY_K)

    thr_copy_b = get_slice(copy_B, threadIdx().x)
    tBgB = partition_S(thr_copy_b, gB)                 # (CPY, CPY_N, CPY_K, k)
    tBsB = partition_D(thr_copy_b, sB)                 # (CPY, CPY_N, CPY_K)

    # mma partition
    thr_mma = get_slice(mma_C, threadIdx().x)
    tCsA = partition_A(thr_mma, sA)                    # (MMA, MMA_M, MMA_K)
    tCsB = partition_B(thr_mma, sB)                    # (MMA, MMA_M, MMA_K)
    tCgC = partition_C(thr_mma, gC)                    # (MMA, MMA_M, MMA_N)

    # accumulator
    tCrC = make_fragment_C(thr_mma, tCgC)
    zeros!(tCrC)

    for k in axes(tAgA, 4)
        copyto!(copy_A, tAsA, view(tAgA, :, :, :, k))
        copyto!(copy_B, tBsB, view(tBgB, :, :, :, k))
        
        cp_async_wait()
        sync_threads()

        @gc_preserve gemm!(mma_C, tCsA, tCsB, tCrC)
        sync_threads()
    end


    copyto!(tCgC, tCrC)
    return nothing
end

function matmul(A, B, C)
    bM = _128
    bN = _128
    bK = _8
    
    sA_layout = make_layout((bM, bK), (_1, bM + _1))
    sB_layout = make_layout((bN, bK), (_1, bN + _1))

    TA = eltype(A)
    TB = eltype(B)
    TC = eltype(C)
	
    copy_A = make_tiled_copy(CopyAtom{CPOP_ASYNC_CACHEALWAYS{TA}, TA}(),
                                    @Layout((32, 8)),
                                    @Layout((4, 1)))
    copy_B = make_tiled_copy(CopyAtom{CPOP_ASYNC_CACHEALWAYS{TB}, TB}(),
                                        @Layout((32, 8)),
                                        @Layout((4, 1)))

    mma_C = make_tiled_mma(UniversalFMA{TA,TB, TC}(), # MMA operation
                           @Layout((16,16)))          # Atom layout

    threads = Int(size(mma_C))
    blocks = (cld(size(A, 1), bM), cld(size(B, 1), bN))

    @cuda threads=threads blocks=blocks matmul_kernel(A, sA_layout, copy_A,
                                                      B, sB_layout, copy_B,
                                                      C, mma_C)
end
```

We can overlap the global-to-shared copies with mma compute.

![](https://developer-blogs.nvidia.com/wp-content/uploads/2020/09/sequence-asynchronous-copy-batches-1.png)

To do this we will explicitly load data from shared memory to registers for
the mma computation and submit a new load from global memory to shared memory for the next tile
before compute.

```julia
function matmul_kernel(A, sA_layout, copy_A,
                       B, sB_layout, copy_B,
                       C, mma_C)
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
    thr_copy_a = get_slice(copy_A, threadIdx().x)      
    tAgA = partition_S(thr_copy_a, gA)                 # (CPY, CPY_M, CPY_K, k)
    tAsA = partition_D(thr_copy_a, sA)                 # (CPY, CPY_M, CPY_K)

    thr_copy_b = get_slice(copy_B, threadIdx().x)
    tBgB = partition_S(thr_copy_b, gB)                 # (CPY, CPY_N, CPY_K, k)
    tBsB = partition_D(thr_copy_b, sB)                 # (CPY, CPY_N, CPY_K)

    # Copy gmem to smem for k_tile=1
    copyto!(copy_A, tAsA, view(tAgA, :, :, :, 1))
    copyto!(copy_B, tBsB, view(tBgB, :, :, :, 1))

    # mma partition
    thr_mma = get_slice(mma_C, threadIdx().x)
    tCsA = partition_A(thr_mma, sA)                    # (MMA, MMA_M, MMA_K)
    tCsB = partition_B(thr_mma, sB)                    # (MMA, MMA_M, MMA_K)
    tCgC = partition_C(thr_mma, gC)                    # (MMA, MMA_M, MMA_N)

    # mma registers
    tCrA = make_fragment_A(mma_C, tCsA)                # (MMA, MMA_M, MMA_K)
    tCrB = make_fragment_B(mma_C, tCsB)                # (MMA, MMA_N, MMA_K)
    tCrC = make_fragment_C(mma_C, tCgC)                # (MMA, MMA_M, MMA_N)
    zeros!(tCrC)

    k_max = size(tAgA, 4)
    for k in 1:k_max
        cp_async_wait()
        sync_threads()

        # copy from smem to rmem
        copyto!(tCrA, tCsA)
        copyto!(tCrB, tCsB)
        sync_threads()

        if k < k_max
            copyto!(copy_A, tAsA, view(tAgA, :, :, :, k+1))
            copyto!(copy_B, tBsB, view(tBgB, :, :, :, k+1))
        end

        @gc_preserve gemm!(mma_C, tCrA, tCrB, tCrC)
    end

    copyto!(tCgC, tCrC)
    return nothing
end
```