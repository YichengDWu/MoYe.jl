# Tensor Cores

Tensor cores are specialized hardware accelerators designed to optimize matrix operations, which are crucial for deep learning and artificial intelligence algorithms.

Incorporating tensor cores can be as straightforward as modifying a single line of code in the existing `mat_mul` function:
```julia
mma = make_tiled_mma(MMAOP_8x8x4_F32F16F16F32_NT(), 
                       @Layout((2,4,1)))
```
!!! note
    The NT in MMAOP_8x8x4_F32F16F16F32_NT indicates that A is in M-major order and B is in N-major order.

## Exploring TiledMMA with Tensor Core Operations
Let's explore what TiledMMA with a tensor core operation entails.
```julia
mma = make_tiled_mma(MMAOP_16x8x8_F32TF32TF32F32_TN(), 
                     @Layout((2,4,1)))
print_typst(mma)
```
![](../assets/tensorcore.svg)

The diagram illustrates the collective loading of data from matrices A and B by threads, according to the specified layout. During the matrix multiply-accumulate (MMA) operation, data is internally shared among threads—this process is seamlessly managed and not visible to the user. Post computation, each thread stores the results as defined by the layout of matrix C, as shown in the illustration.

## LDMatrix

The `ldmatrix` instruction at the warp level facilitates the loading of data from shared memory into registers and rearranges them to align with a tensor core MMA operation.

Given a tensor core MMA operation, the shuffling can be "inverted" to obtain a TiledCopy count for the shuffling.
```julia
smem_copy_A = make_tiled_copy_A(CopyAtom{LDSM_U32x4_N, Float32}(), mma)
print_typst(smem_copy_A)
```
![](../assets/smem_copy_A.svg)

The resulting layout matches the layout of A in the mma.

!!! note
    The TN in MMAOP_16x8x8_F32TF32TF32F32_TN specifies that both A and B are in K-major order.
    The N in LDSM_U32x4_N also indicates K-major order.

!!! note 
    The `ldmatrix` requires four consecutive threads to load 16 consecutive bytes, demanding that the layout of A in shared memory meet this specification.

For B:
```julia
smem_copy_B = make_tiled_copy_B(CopyAtom{LDSM_U32x2_N, Float32}(), mma)
print_typst(smem_copy_B)
```

However, using LDSM_U32x4_N for `B` would not be compatible with its layout in mma.

Another configuration:
```julia
mma = make_tiled_mma(MMAOP_16x8x8_F32TF32TF32F32_TN(), 
                            @Layout((2,2,1)),
                            (_32,_32,_8))
smem_copy_A = make_tiled_copy_A(CopyAtom{LDSM_U32x4_N, Float32}(), mma)
smem_copy_B = make_tiled_copy_B(CopyAtom{LDSM_U32x4_N, Float32}(), mma)
```

We then use `smem_copy_A` and `smem_copy_B` to re-tile the shared memory and registers
```julia
smem_thr_copy_A = get_slice(smem_copy_A, threadIdx().x)   
tCsA            = partition_S(smem_thr_copy_A, sA)                
tCrA_copy_view  = retile_D(smem_thr_copy_A, tCrA)

smem_thr_copy_B = get_slice(smem_copy_B, threadIdx().x)   
tCsB            = partition_S(smem_thr_copy_B, sB)                
tCrB_copy_view  = retile_D(smem_thr_copy_B, tCrB)
```

Here, `retile_D` acts as a composition of `tCrA` with the partitioner `smem_thr_copy_A`.

## MatMul

This example computes C = A * B, with A in M-major and B in K-major order.

```julia
@views function matmul_kernel(A, sA_layout, gmem_copy_A, smem_copy_A,
                              B, sB_layout, gmem_copy_B, smem_copy_B,
                              C, mma)
    M = size(A, 1)
    N = size(B, 1)
    K = size(A, 2)

    bM = size(sA_layout, 1)
    bN = size(sB_layout, 1)
    bK = size(sB_layout, 2)

    sA = MoYeSharedArray(eltype(A), sA_layout)        # (bM, bK, 2)
    sB = MoYeSharedArray(eltype(B), sB_layout)        # (bN, bK, 2)

    mA = MoYeArray(A)
    mB = MoYeArray(B)
    mC = MoYeArray(C)

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
    thr_mma = get_slice(mma, threadIdx().x)
    tCgC = partition_C(thr_mma, gC)                    # (MMA, MMA_M, MMA_N)

    # mma registers
    tCrA = partition_fragment_A(thr_mma, sA[:, :, 1])    # (MMA, MMA_M, MMA_K)
    tCrB = partition_fragment_B(thr_mma, sB[:, :, 1])    # (MMA, MMA_N, MMA_K)
    tCrC = make_fragment_C(thr_mma, tCgC)                # (MMA, MMA_M, MMA_N)
    zeros!(tCrC)

    # retile 
    smem_thr_copy_A = get_slice(smem_copy_A, threadIdx().x)   
    tCsA            = partition_S(smem_thr_copy_A, sA)   # (MMA, MMA_M, MMA_K, 2)      
    tCrA_copy_view  = retile_D(smem_thr_copy_A, tCrA)    # (MMA, MMA_M, MMA_K)    

    smem_thr_copy_B = get_slice(smem_copy_B, threadIdx().x)   
    tCsB            = partition_S(smem_thr_copy_B, sB)  # (MMA, MMA_N, MMA_K, 2) 
    tCrB_copy_view  = retile_D(smem_thr_copy_B, tCrB)   # (MMA, MMA_N, MMA_K) 

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
    for k_tile in 1:k_tile_max
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
            
            @gc_preserve gemm!(mma, tCrA[:, :, k_block], tCrB[:, :, k_block], tCrC)
        end
    end

    copyto!(tCgC, tCrC)
    return nothing
end

function matmul(A, B, C)
    bM = _128
    bN = _128
    bK = _32
    
    sA_layout = make_layout((bM, bK, _2), (_1, bM, bM * bK))    # M-major
    sB_layout = make_layout((bN, bK, _2), (bK, _1, bN * bK))    # K-major

    TA = eltype(A)
    TB = eltype(B)
    TC = eltype(C)
	
    gmem_copy_A = make_tiled_copy(CopyAtom{UniversalCopy{UInt128}, TA}(),
                                  @Layout((16, 8)),
                                  @Layout((4, 1)))
    gmem_copy_B = make_tiled_copy(CopyAtom{UniversalCopy{UInt128}, TB}(),
                                  @Layout((16, 8), (8, 1)),
                                  @Layout((1, 4)))

    mma = make_tiled_mma(MMAOP_16x8x8_F32TF32TF32F32_TN(), 
                         @Layout((2,2,1)),
                         (_32,_32,_8))

    # A is M-major so we cannot use LDSM_U32x4_N 
    smem_copy_A = make_tiled_copy_A(CopyAtom{UniversalCopy{TA}, TA}(), mma)
    smem_copy_B = make_tiled_copy_B(CopyAtom{LDSM_U32x4_N, TB}(), mma)

    threads = Int(size(mma))
    blocks = (cld(size(A, 1), bM), cld(size(B, 1), bN))

    @cuda threads=threads blocks=blocks matmul_kernel(A, sA_layout, gmem_copy_A, smem_copy_A,
                                                      B, sB_layout, gmem_copy_B, smem_copy_B,
                                                      C, mma)
end


function test()
    A = CUDA.randn(Float32, 2048, 256)   # M-major
    B = CUDA.randn(Float32, 256, 2048)
    B_T =  B'  # K-major
    C =  CUDA.randn(Float32, 2048, 2048)
    matmul(A, B_T, C)
    CUDA.synchronize()
    @test C == A * B
    CUDA.unsafe_free!(A)
    CUDA.unsafe_free!(B)
    CUDA.unsafe_free!(C)
end

test()
```