using MoYe, CUDA, Test
using MoYe: @loopinfo

const X = MoYe.One()

function matmul_kernel(A, blocklayout_A, threadlayout_A, B, blocklayout_B, threadlayout_B,
                       C, blocklayout_C, threadlayout_C)
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

    blocktile_A = @tile mA (bM, bK) (blockIdx().x, :) # (bM,bK,k)
    blocktile_B = @tile mB (bN, bK) (blockIdx().y, :) # (bN,bK,k)
    blocktile_C = @tile mC (bM, bN) (blockIdx().x, blockIdx().y) # (bM,bN)

    # Load data A and B from gmem to smem a and b
    threadtile_gA = @parallelize blocktile_A threadlayout_A threadIdx().x # (tM,tK,k)
    threadtile_sA = @parallelize sA threadlayout_A threadIdx().x # (tM,tK）

    threadtile_gB = @parallelize blocktile_B threadlayout_B threadIdx().x # (tN,tK,k)
    threadtile_sB = @parallelize sB threadlayout_B threadIdx().x # (tN,tK）

    # For mma computation
    computetile_sA = @parallelize sA threadlayout_C threadIdx().x (X, :)
    computetile_sB = @parallelize sB threadlayout_C threadIdx().x (:, X)
    computetile_gC = @parallelize blocktile_C threadlayout_C threadIdx().x # (128 ÷ 16, 128 ÷ 16)

    frg_c = make_fragment_like(computetile_gC)
    zeros!(frg_c)

    k_max = size(threadtile_gA, 3)

    for i in 1:k_max
        # copy gmem to smem
        copyto!(threadtile_sA, view(threadtile_gA, :, :, i))
        copyto!(threadtile_sB, view(threadtile_gB, :, :, i))
        cp_async_wait()
        sync_threads()

        # classic three nested for loops
        for k in axes(computetile_sA, 2)
            @loopinfo unroll for m in axes(computetile_sA, 1)
                @loopinfo unroll for n in axes(computetile_sB, 1)
                    @inbounds frg_c[m, n] += computetile_sA[m, k] * computetile_sB[n, k]
                end
            end
        end

        sync_threads()
    end

    copyto!(computetile_gC, frg_c)
    return nothing
end

function matmul(A, B, C)
    M = size(A, 1)
    N = size(B, 1)
    K = size(A, 2)

    blocklayout_A = @Layout (128, 8)
    blocklayout_B = @Layout (128, 8)
    blocklayout_C = @Layout (128, 128)

    threadlayout_A = @Layout (32, 8)
    threadlayout_B = @Layout (32, 8)
    threadlayout_C = @Layout (32, 8)

    threads = Int(size(threadlayout_C))

    bM = size(blocklayout_A, 1)
    bN = size(blocklayout_B, 1)

    blocks = (cld(M, bM), cld(N, bN))

    @cuda threads=threads blocks=blocks matmul_kernel(A, blocklayout_A, threadlayout_A, B,
                                                      blocklayout_B, threadlayout_B, C,
                                                      blocklayout_C, threadlayout_C)
end

function test()
    A = CUDA.randn(Float32, 2048, 256)
    B = CUDA.randn(Float32, 2048, 256)
    C = CUDA.randn(Float32, 2048, 2048)
    matmul(A, B, C)
    @test C == A * B'
    CUDA.unsafe_free!(A)
    CUDA.unsafe_free!(B)
    CUDA.unsafe_free!(C)
end
