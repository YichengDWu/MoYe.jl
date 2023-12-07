
using MoYe, CUDA, Test

function matmul_kernel(A, blocklayout_A, B, blocklayout_B, C, tiled_copy, tiled_mma)
    sA = MoYeSharedArray(eltype(A), blocklayout_A)
    sB = MoYeSharedArray(eltype(B), blocklayout_B)

    M = size(A, 1)
    N = size(B, 1)
    K = size(A, 2)

    mA = MoYeArray(A, (M, K))
    mB = MoYeArray(B, (N, K))
    mC = MoYeArray(C, (M, N))

    bM = size(blocklayout_A, 1)
    bN = size(blocklayout_B, 1)
    bK = size(blocklayout_B, 2)

    blk_A = @tile mA (bM, bK) (blockIdx().x, :) # (bM,bK,k)
    blk_B = @tile mB (bN, bK) (blockIdx().y, :) # (bN,bK,k)
    blk_C = @tile mC (bM, bN) (blockIdx().x, blockIdx().y) # (bM,bN)

    # For mma computation
    thread_idx = Int(threadIdx().x)
    thr_mma = get_thread_slice(tiled_mma, thread_idx)
    thr_A = partition_A(thr_mma, sA)
    thr_B = partition_B(thr_mma, sB)
    thr_C = partition_C(thr_mma, blk_C)

    frg_c = make_fragment_like(thr_C)
    zeros!(frg_c)

    for i in axes(blk_A, 3)
        # copy gmem to smem
        @collective copyto!(tiled_copy, sA, view(blk_A, :, :, i))
        @collective copyto!(tiled_copy, sB, view(blk_B, :, :, i))

        cp_async_wait()
        sync_threads()

        MoYe.gemm!(tiled_mma, view(frg_c, :), thr_A, thr_B, view(frg_c, :))
        sync_threads()
    end
    # copy rmem to gmem
    copyto!(thr_C, frg_c)
    return nothing
end

function matmul_universal(A, B, C)
    M = size(A, 1)
    N = size(B, 1)

    blocklayout_A = @Layout (128, 8)
    blocklayout_B = @Layout (128, 8)

    tiled_mma = MoYe.make_tiled_mma(UniversalFMA{Float32, Float32, Float32, Float32}(), @Layout((32,8)))
    tiled_copy = make_tiled_copy(CopyAtom{MoYe.CPOP_ASYNC_CACHEALWAYS{UInt128, UInt128}, Float32}(), @Layout((32,8)), @Layout((4,1)))
    threads = Int(size(tiled_copy))

    bM = size(blocklayout_A, 1)
    bN = size(blocklayout_B, 1)

    blocks = (cld(M, bM), cld(N, bN))

    @cuda threads=threads blocks=blocks matmul_kernel(A, blocklayout_A,
                                                      B, blocklayout_B,
                                                      C, tiled_copy, tiled_mma)
end

function test_gemm_universal()
    A = CUDA.randn(Float32, 2048, 256)
    B = CUDA.randn(Float32, 2048, 256)
    C = CUDA.randn(Float32, 2048, 2048)
    matmul_universal(A, B, C)
    CUDA.synchronize()
    @test C == A * B'
    CUDA.unsafe_free!(A)
    CUDA.unsafe_free!(B)
    CUDA.unsafe_free!(C)
end


function test_gemm_tensorcore()
    A = CUDA.randn(Float16, 2048, 256)
    B = CUDA.randn(Float16, 2048, 256)
    C = CUDA.randn(Float16, 2048, 2048)

    M = size(A, 1)
    N = size(B, 1)

    blocklayout_A = @Layout (128, 8) GenRowMajor
    blocklayout_B = @Layout (128, 8) GenRowMajor

    tiled_mma = MoYe.make_tiled_mma(MMAOP_16x8x8_F16F16F16F16_TN(), @Layout((8,1)))
    tiled_copy = make_tiled_copy(CopyAtom{UniversalCopy{Float16, Float16}, Float16}(), @Layout((32,8)))
    threads = Int(size(tiled_copy))

    bM = size(blocklayout_A, 1)
    bN = size(blocklayout_B, 1)

    blocks = (cld(M, bM), cld(N, bN))

    @cuda threads=threads blocks=blocks matmul_kernel(A, blocklayout_A,
                                                    B, blocklayout_B,
                                                    C, tiled_copy, tiled_mma)
    @test  C == A*B'
end
