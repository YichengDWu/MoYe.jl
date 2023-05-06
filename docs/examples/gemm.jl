function gemme_kernel(A, blocklayout_A, threadlayout_A,
                      B, blocklayout_B, threadlayout_B,
                      C, blocklayout_C, threadlayout_C)
    a = MoYeSharedArray(eltype(A), blocklayout_A)
    b = MoYeSharedArray(eltype(B), blocklayout_B)

    M = size(A, 1)
    N = size(B, 1)
    K = size(A, 2)

    moye_A = moye_Array(pointer(A), (M, K))
    moye_B = moye_Array(pointer(B), (N, K))
    moye_C = moye_Array(pointer(C), (M, N))

    bM = size(blocklayout_A, 1)
    bN = size(blocklayout_B, 1)
    bK = size(blocklayout_B, 2)

    blocktile_A = @tile moye_A (bM, bK) (blockIdx().x, :) # (bM,bK,k)
    blocktile_B = @tile moye_B (bN, bK) (blockIdx().y, :) # (bN,bK,k)
    blocktile_C = @tile moye_C (bM, bN) (blockIdx().x, blockIdx().y) # (bM,bN)

    # Load data A and B from gmem to smem a and b
    threadtile_A = @parallize blocktile_A threadlayout_A threadIdx().x # (tM,tK,k)
    threadtile_a = @parallize a threadlayout_A threadIdx().x # (tM,tK）

    threadtile_B = @parallize blocktile_B threadlayout_B threadIdx().x # (tN,tK,k)
    threadtile_b = @parallize b threadlayout_B threadIdx().x # (tN,tK）

    # For mma computation
    computetile_A = @parallize a threadlayout_C threadIdx().x (One(), :)
    computetile_B = @parallize b threadlayout_C threadIdx().x (:, One())
    computetile_C = @parallize blocktile_C threadlayout_C threadIdx().x # (tM, tN)

    frag_c = make_fragment_like(computetile_C)
    zeros!(frag_c)

    k_max = size(threadtile_a, 2)

    for k in 1:k_max
        # copy gmem to smem
        cucopyto!(threadtile_a, view(threadtile_A, (:, :, k)))
        cucopyto!(threadtile_b, view(threadtile_B, (:, :, k)))
        cp_async_wait()
        sync_threads()
        gemm!(computetile_A, computetile_B, frag_c)
        sync_threads()
    end
end

function gemm(A, B, C)
    M = size(A, 1)
    N = size(B, 1)
    K = size(A, 2)

    blocklayout_A = @Layout (128, 8)
    blocklayout_B = @Layout (128, 8)
    blocklayout_C = @Layout (128, 128)

    threadlayout_A = @Layout (32, 8)
    threadlayout_B = @Layout (32, 8)
    threadlayout_C = @Layout (16, 16)

    threads = dynamic(size(threadlayout_C))

    bM = size(blocklayout_A, 1)
    bN = size(blocklayout_B, 1)

    blocks = (cld(M, bM), cld(N, bN))

    @cuda threads=threads blocks=blocks gemme_kernel(A, strideA, blocklayout_A, threadlayout_A,
                                                     B, strideB, blocklayout_B, threadlayout_B,
                                                     C, strideC, blocklayout_C, threadlayout_C)
end
