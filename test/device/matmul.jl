using MoYe, CUDA, Test

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

        @gc_preserve gemm!(tCrC, tCsA, tCsB, tCrC)
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

    tA = @Layout (32, 8)
    tB = @Layout (32, 8)
    tC = @Layout (16, 16)

    threads = Int(size(tC))
    blocks = (cld(size(A, 1), bM), cld(size(B, 1), bN))

    @cuda threads=threads blocks=blocks matmul_kernel(A, sA_layout, tA,
                                                      B, sB_layout, tB,
                                                      C, tC)
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

