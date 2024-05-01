using MoYe, CUDA, Test

@views function matmul_kernel(A, sA_layout, gmem_copy_A, smem_copy_A,
                       B, sB_layout, gmem_copy_B, smem_copy_B,
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

    # gmem copy partition
    gmem_thr_copy_a = get_slice(gmem_copy_A, threadIdx().x)      
    tAgA = partition_S(gmem_thr_copy_a, gA)                 # (CPY, CPY_M, CPY_K, k)
    tAsA = partition_D(gmem_thr_copy_a, sA)                 # (CPY, CPY_M, CPY_K)

    gmem_thr_copy_b = get_slice(gmem_copy_B, threadIdx().x)
    tBgB = partition_S(gmem_thr_copy_b, gB)                 # (CPY, CPY_N, CPY_K, k)
    tBsB = partition_D(gmem_thr_copy_b, sB)                 # (CPY, CPY_N, CPY_K)

    # mma partition
    thr_mma = get_slice(mma_C, threadIdx().x)
    tCsA = partition_A(thr_mma, sA)                    # (MMA, MMA_M, MMA_K)
    tCsB = partition_B(thr_mma, sB)                    # (MMA, MMA_M, MMA_K)
    tCgC = partition_C(thr_mma, gC)                    # (MMA, MMA_M, MMA_N)

    tCrA = make_fragment_A(thr_mma, tCsA)              # (MMA, MMA_M, MMA_K)
    tCrB = make_fragment_B(thr_mma, tCsB)              # (MMA, MMA_N, MMA_K)
    tCrC = make_fragment_C(thr_mma, tCgC)              # (MMA, MMA_M, MMA_N)
    zeros!(tCrC)

    # retile 
    smem_thr_copy_A = get_slice(smem_copy_A, threadIdx().x)
    smem_thr_copy_B = get_slice(smem_copy_B, threadIdx().x)
    tCsA_retiled = partition_S(smem_thr_copy_A, sA)
    tCsB_retiled = partition_S(smem_thr_copy_B, sB)
    tCrA_retiled = retile_D(smem_thr_copy_A, tCrA)
    tCrB_retiled = retile_D(smem_thr_copy_B, tCrB)
    
    
    k_tile_max = size(tAgA, 4)
    @cuprintln "k_tile_max: $k_tile_max"
    k_tile = 1
   # for k_tile in 1:k_tile_max
            # copy from global to shared
        copyto!(gmem_copy_A, tAsA, view(tAgA, :, :, :, k_tile))
        copyto!(gmem_copy_B, tBsB, view(tBgB, :, :, :, k_tile))
        cp_async_wait()
        sync_threads()

        # copy from shared to registers
        copyto!(smem_copy_A, tCrA_retiled, tCsA_retiled)
        copyto!(smem_copy_B, tCrB_retiled, tCsB_retiled)
      #  MoYe.copyto_unpack!(MoYe.CopyTraits{LDSM_U32x4_N}(), view(tCrB_retiled, (:,_1), _1, _1), view(tCsB_retiled, (:,_1), _1, _1))
        if threadIdx().x == 1
            @cuprintln "Thread 1"

            @cuprintln Int32(sB[17,1]), Int32(sB[17,2]), Int32(sB[17,3]), Int32(sB[17,4]), Int32(sB[17,5]), Int32(sB[17,6]), Int32(sB[17,7]), Int32(sB[17,8]), Int32(sB[17,9]), Int32(sB[17,10]), Int32(sB[17,11]), Int32(sB[17,12]), Int32(sB[17,13]), Int32(sB[17,14]), Int32(sB[17,15]), Int32(sB[17,16])
            @cuprintln Int32(sB[18,1]), Int32(sB[18,2]), Int32(sB[18,3]), Int32(sB[18,4]), Int32(sB[18,5]), Int32(sB[18,6]), Int32(sB[18,7]), Int32(sB[18,8]), Int32(sB[18,9]), Int32(sB[18,10]), Int32(sB[18,11]), Int32(sB[18,12]), Int32(sB[18,13]), Int32(sB[18,14]), Int32(sB[18,15]), Int32(sB[18,16])
            
            @cuprintln Int32(tCrB[1,1,1]), Int32(tCrB[2,1,1]), Int32(tCrB[1,2,1]), Int32(tCrB[2,2,1])
            @cuprintln Int32(tCrB_retiled[1,1,1]), Int32(tCrB_retiled[2,1,1]), Int32(tCrB_retiled[3,1,1]), Int32(tCrB_retiled[4,1,1])
        end
        if threadIdx().x == 25
            @cuprintln "Thread 25"
            @cuprintln Int32(tCsB_retiled[1]), Int32(tCsB_retiled[2]), Int32(tCsB_retiled[3]), Int32(tCsB_retiled[4])
        end
        @gc_preserve gemm!(mma_C, tCrC, tCrA, tCrB, tCrC)


        @inbounds tCrC[1]  # compiler bug, have to load after copyto!

        sync_threads()
#    end
    
    copyto!(tCgC, tCrC) 
    @inbounds tCrC[1]  # compiler bug, have to load after copyto!

    sync_threads()
    return nothing
end


function matmul(A, B, C)
    bM = _32
    bN = _32
    bK = _16

    sA_atom_layout = @Layout (32, 8) (1, 32)
    sB_atom_layout = @Layout (8, 16) (16, 1)
    
    sA_layout = MoYe.tile_to_shape(sA_atom_layout, (bM, bK))
    sB_layout = MoYe.tile_to_shape(sB_atom_layout, (bN, bK))

    TA = eltype(A)
    TB = eltype(B)
    TC = eltype(C)
	
    gmem_copy_A = make_tiled_copy(CopyAtom{CPOP_ASYNC_CACHEALWAYS{UInt128}, TA}(),
                                  @Layout((4, 8)),
                                  @Layout((4, 1)))
    gmem_copy_B = make_tiled_copy(CopyAtom{CPOP_ASYNC_CACHEALWAYS{UInt128}, TB}(),
                                  @Layout((8, 4), (4, 1)),
                                  @Layout((1, 4)))

    mma = make_tiled_mma(MMAOP_16x8x8_F32TF32TF32F32_TN(),
                           @Layout((1,2,1)),
                           (_32, _32, _8))

    # Note: A is M-major so we can only use `UniversalCopy`
    smem_copy_A =  make_tiled_copy_A(CopyAtom{UniversalCopy{TA}, TA}(), mma)
    smem_copy_B =  make_tiled_copy_B(CopyAtom{LDSM_U32x4_N, TB}(), mma)

    threads = Int(size(mma))
    blocks = (cld(size(A, 1), bM), cld(size(B, 1), bN))

    @cuda threads=threads blocks=blocks matmul_kernel(A, sA_layout, gmem_copy_A, smem_copy_A,
                                                      B, sB_layout, gmem_copy_B, smem_copy_B,
                                                      C, mma)
end

function test()
    M = 32
    K = 16
    N = 32
    A = CuArray(reshape(collect(1:M*K) .* 1f0, (M,K)))   
    B = CuArray(reshape(collect(1:N*K) .* 1f0, (K,N)))     # K-major
    C = CuArray(ones(Float32, (M,N)))
    matmul(A, B', C)
    CUDA.synchronize()
    @test C == A * B
    CUDA.unsafe_free!(A)
    CUDA.unsafe_free!(B)
    CUDA.unsafe_free!(C)
end

test()