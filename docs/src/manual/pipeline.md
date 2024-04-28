## Overlap global-to-shared copies with mma compute

We can overlap global-to-shared memory copies with mma compute on registers.

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
    tCrA = make_fragment_A(thr_mma, tCsA)                # (MMA, MMA_M, MMA_K)
    tCrB = make_fragment_B(thr_mma, tCsB)                # (MMA, MMA_N, MMA_K)
    tCrC = make_fragment_C(thr_mma, tCgC)                # (MMA, MMA_M, MMA_N)
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