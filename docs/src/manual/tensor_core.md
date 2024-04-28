# Tensor Cores

Tensor cores are specialized hardware accelerators designed to optimize matrix operations, which are crucial for deep learning and artificial intelligence algorithms.

Switching to tensor cores can be as simple as modifying just one line of code in the previous matmul function:

```julia
mma_C = make_tiled_mma(MMAOP_16x8x8_F32TF32TF32F32_TN(), 
                       @Layout((2,4,1)))
```

Let's explore what TiledMMA entails.
```julia
print_typst(mma_C)
```
![](../assets/tensorcore.svg)

At first glance, the diagram may seem complex, but the concept is straightforward: the threads collective load data from matrices A and B according to the specified layout. During the matrix multiply-accumulate (MMA) computation, data is internally shared among threads—a process that is not transparent to the user. Once the computation is complete, each thread stores the results as dictated by the layout of matrix C shown in the illustration.

Of course you can also choose other mma atoms. They just work.
```julia
mma_C = make_tiled_mma(MMAOP_16x8x8_F32F16F16F32_TN(), 
                                @Layout((2,4,1)))
```

## LDMatrix

(explaination)

```julia
@views function matmul_kernel(A, sA_layout, gmem_copy_A, smem_copy_A,
                              B, sB_layout, gmem_copy_B, smem_copy_B,
                              C, mma_C)
    M = size(A, 1)
    N = size(B, 1)
    K = size(A, 2)

    bM = size(sA_layout, 1)
    bN = size(sB_layout, 1)
    bK = size(sB_layout, 2)

    sA = MoYeSharedArray(eltype(A), sA_layout)        # (bM, bK, 2)
    sB = MoYeSharedArray(eltype(B), sB_layout)        # (bN, bK, 2)

    mA = MoYeArray(A, (M, K))
    mB = MoYeArray(B, (N, K))
    mC = MoYeArray(C, (M, N))

    gA = @tile mA (bM, bK) (blockIdx().x, :)
    gB = @tile mB (bN, bK) (blockIdx().y, :)
    gC = @tile mC (bM, bN) (blockIdx().x, blockIdx().y)

    # gmem copy partition
    gmem_thr_copy_A = get_slice(gmem_copy_A, threadIdx().x)      
    tAgA = partition_S(gmem_thr_copy_A, gA)                 # (CPY, CPY_M, CPY_K, k)
    tAsA = partition_D(gmem_thr_copy_A, sA)                 # (CPY, CPY_M, CPY_K, 2)

    gmem_thr_copy_B = get_slice(gmem_copy_B, threadIdx().x)
    tBgB = partition_S(gmem_thr_copy_B, gB)                 # (CPY, CPY_N, CPY_K, k)
    tBsB = partition_D(gmem_thr_copy_B, sB)                 # (CPY, CPY_N, CPY_K, 2)

    # Copy gmem to smem for k_tile=1
    copyto!(gmem_copy_A, tAsA[:, :, :, 1], tAgA[:, :, :, 1])
    copyto!(gmem_copy_B, tBsB[:, :, :, 1], tBgB[:, :, :, 1])

    # mma partition
    thr_mma = get_slice(mma_C, threadIdx().x)
    tCrA = partition_fragment_A(thr_mma, sA[:,:,1])
    tCrB = partition_fragment_B(thr_mma, sB[:,:,1])
    tCgC = partition_C(thr_mma, gC)  
    tCrC = make_fragment_C(thr_mma, tCgC)              
    zeros!(tCrC)


    smem_thr_copy_A = get_slice(smem_copy_A, threadIdx().x)   
    tCsA            = partition_S(smem_thr_copy_A, sA)                
    tCrA_copy_view  = retile_D(smem_thr_copy_A, tCrA)

    smem_thr_copy_B = get_slice(smem_copy_B, threadIdx().x)   
    tCsB            = partition_S(smem_thr_copy_B, sB)  
    tCrB_copy_view  = retile_D(smem_thr_copy_B, tCrB)

    tCrA_copy_view = retile_D(smem_thr_copy_A, tCrA)
    tCrB_copy_view = retile_D(smem_thr_copy_B, tCrB)

    cp_async_wait()
    sync_threads()

    # Copy smem to rmem for k_block=1
    smem_read = 1
    smem_write = 2
    tCsA_p = view(tCsA, :, :, :, smem_read)
    tCsB_p = view(tCsB, :, :, :, smem_read)
    copyto!(smem_copy_A, tCrA_copy_view[:, :, 1], tCsA_p[:, :, 1])
    copyto!(smem_copy_B, tCrB_copy_view[:, :, 1], tCsB_p[:, :, 1])

    k_tile_max = size(tAgA, 4)
    k_block_max = static_size(tCrA, 3)
   #= for k_tile in 1:k_tile_max
        @loopinfo unroll for k_block in _1:k_block_max
            k_block_next = k_block + 1 
            if k_block == k_block_max
                cp_async_wait()
                sync_threads()
                tCsA_p = view(tCsA, :, :, :, smem_read)
                tCsB_p = view(tCsB, :, :, :, smem_read)
                k_block_next = 1
            end
            
            copyto!(smem_copy_A, tCrA_copy_view[:, :, k_block_next], tCsA_p[:, :, k_block_next])
            copyto!(smem_copy_B, tCrB_copy_view[:, :, k_block_next], tCsB_p[:, :, k_block_next])   

            if k_block == _1 && k_tile<k_tile_max
                copyto!(gmem_copy_A, tAsA[:, :, :, smem_write], tAgA[:, :, :, k_tile+1])
                copyto!(gmem_copy_B, tBsB[:, :, :, smem_write], tBgB[:, :, :, k_tile+1])
                smem_read, smem_write = smem_write, smem_read
            end
            
            @gc_preserve gemm!(mma_C, tCrA[:, :, k_block], tCrB[:, :, k_block], tCrC)
        end
    end
=#
    copyto!(tCgC, tCrC)
    return nothing
end

function matmul(A, B, C)
    bM = _128
    bN = _128
    bK = _8
    
    sA_layout = make_layout((bM, bK, _2), (_1, bM + _2, (bM + _2) * bK))
    sB_layout = make_layout((bN, bK, _2), (_1, bN + _2, (bN + _2) * bK))

    TA = eltype(A)
    TB = eltype(B)
    TC = eltype(C)
	
    gmem_copy_A = make_tiled_copy(CopyAtom{CPOP_ASYNC_CACHEALWAYS{Float64}, TA}(),
                             @Layout((32, 8)),
                             @Layout((2, 1)))
    gmem_copy_B = make_tiled_copy(CopyAtom{CPOP_ASYNC_CACHEALWAYS{Float64}, TB}(),
                             @Layout((32, 8)),
                             @Layout((2, 1)))

    mma_C = make_tiled_mma(MMAOP_16x8x8_F32TF32TF32F32_TN(),
                              @Layout((2,4,1)))

    smem_copy_A = make_tiled_copy_A(CopyAtom{LDSM_U32x4_N, TA}(), mma_C)
    smem_copy_B = make_tiled_copy_B(CopyAtom{LDSM_U32x2_N, TB}(), mma_C)

    threads = Int(size(mma_C))
    blocks = (cld(size(A, 1), bM), cld(size(B, 1), bN))

    @cuda threads=threads blocks=blocks matmul_kernel(A, sA_layout, gmem_copy_A, smem_copy_A,
                                                      B, sB_layout, gmem_copy_B, smem_copy_B,
                                                      C, mma_C)
end
```