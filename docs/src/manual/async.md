# Asynchronous Memory Copies

The NVIDIA Ampere architecture introduced `memcpy_async`, a feature that allows for asynchronous data copies between GPU global and shared memory. This frees up threads from managing data movement, allowing them to focus on computation.

To use this feature, we change the `TiledCopy` to the following:
```julia
copy_A = make_tiled_copy(CopyAtom{CPOP_ASYNC_CACHEALWAYS{TA}, TA}(),
                                @Layout((32, 8)),
                                @Layout((1, 1)))
copy_B = make_tiled_copy(CopyAtom{CPOP_ASYNC_CACHEALWAYS{TB}, TB}(),
                                    @Layout((32, 8)),
                                    @Layout((1, 1)))
```

## Updated Kernel

```julia
function matmul_kernel(A, sA_layout, copy_A,
                       B, sB_layout, copy_B,
                       C, mma_C)
    sA = MoYeSharedArray(eltype(A), sA_layout)
    sB = MoYeSharedArray(eltype(B), sB_layout)

    mA = MoYeArray(A)
    mB = MoYeArray(B)
    mC = MoYeArray(C)

    bM = size(sA_layout, 1)
    bN = size(sB_layout, 1)
    bK = size(sB_layout, 2)

    gA = @tile mA (bM, bK) (blockIdx().x, :)
    gB = @tile mB (bN, bK) (blockIdx().y, :)
    gC = @tile mC (bM, bN) (blockIdx().x, blockIdx().y)

    # Copy partition
    thr_copy_a = get_slice(copy_A, threadIdx().x)      
    tAgA = partition_S(thr_copy_a, gA)                 # (CPY, CPY_M, CPY_K, k)
    tAsA = partition_D(thr_copy_a, sA)                 # (CPY, CPY_M, CPY_K)

    thr_copy_b = get_slice(copy_B, threadIdx().x)
    tBgB = partition_S(thr_copy_b, gB)                 # (CPY, CPY_N, CPY_K, k)
    tBsB = partition_D(thr_copy_b, sB)                 # (CPY, CPY_N, CPY_K)

    # MMA partition
    thr_mma = get_slice(mma_C, threadIdx().x)
    tCsA = partition_A(thr_mma, sA)                    # (MMA, MMA_M, MMA_K)
    tCsB = partition_B(thr_mma, sB)                    # (MMA, MMA_M, MMA_K)
    tCgC = partition_C(thr_mma, gC)                    # (MMA, MMA_M, MMA_N)

    # Accumulator
    tCrC = make_fragment_C(thr_mma, tCgC)
    zeros!(tCrC)

    for k in axes(tAgA, 4)
        copyto!(copy_A, tAsA, view(tAgA, :, :, :, k))
        copyto!(copy_B, tBsB, view(tBgB, :, :, :, k))
        
        cp_async_wait()

        @gc_preserve gemm!(mma_C, tCrC, tCsA, tCsB, tCrC)
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
                                    @Layout((1, 1)))
    copy_B = make_tiled_copy(CopyAtom{CPOP_ASYNC_CACHEALWAYS{TB}, TB}(),
                                        @Layout((32, 8)),
                                        @Layout((1, 1)))

    mma_C = make_tiled_mma(UniversalFMA{TA,TB, TC}(), # MMA operation
                           @Layout((32,8)))          # Atom layout

    threads = Int(size(mma_C))
    blocks = (cld(size(A, 1), bM), cld(size(B, 1), bN))

    @cuda threads=threads blocks=blocks matmul_kernel(A, sA_layout, copy_A,
                                                      B, sB_layout, copy_B,
                                                      C, mma_C)
end


function test()
    A =  CUDA.randn(Float32, 2048, 256)
    B =  CUDA.randn(Float32, 2048, 256)
    C =  CUDA.randn(Float32, 2048, 2048)
    matmul(A, B, C)
    CUDA.synchronize()
    @test C == A * B'
    CUDA.unsafe_free!(A)
    CUDA.unsafe_free!(B)
    CUDA.unsafe_free!(C)
end

test()
```

## Vectorized Copy

We can enable vectorized copies from global to shared memory by changing `CPOP_ASYNC_CACHEALWAYS{TA}` and `CPOP_ASYNC_CACHEALWAYS{TB}` to `CPOP_ASYNC_CACHEALWAYS{Float64}`. However, this will result in a memory misalignment error because we padded `sA` and `sB` by one row. The element at `[1,2]` is not aligned to 8 bytes as required by the `copy_async` instruction.

To fix this, we need to adjust the padding:
```julia
sA_layout = make_layout((bM, bK), (_1, bM + _2))
sB_layout = make_layout((bN, bK), (_1, bN + _2))
```
