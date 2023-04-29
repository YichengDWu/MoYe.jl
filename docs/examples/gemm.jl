function gemme_kernel(M, N, K,
                      A, strideA, blocklayoutA, threadlayoutA,
                      B, strideB, blocklayoutB, threadlayoutB,
                      C, strideC, blocklayoutC, threadlayoutC)

    shmemA = Shambles.SharedMemory(eltype(A), cosize(blocklayoutA))
    shmemB = Shambles.SharedMemory(eltype(B), cosize(blocklayoutB))
    a = CuTeArray(shmemA, blocklayoutA)
    b = CuTeArray(shmemB, blocklayoutB)

    cuteA = CuTeArray(pointer(A), (M, K), strideA)
    cuteB = CuTeArray(pointer(B), (N, K), strideB)
    cuteC = CuTeArray(pointer(C), (M, N), strideC)

    bM = size(blocklayoutA, 1)
    bN = size(blocklayoutB, 1)
    bK = size(blocklayoutB, 2)

    blocktileA = local_tile(cuteA, (bM, bK), (blockIdx().x, :)) # (BLK_M,BLK_K,k)
    blocktileB = local_tile(cuteB, (bN, bK), (blockIdx().y, :)) # (BLK_N,BLK_K,k)
    blocktileC = local_tile(cuteC, (bM, bN), (blockIdx().x, blockIdx().y)) # (BLK_M,BLK_N,k)

    threadtileA = local_partition(blocktileA, threadlayoutA, threadIdx().x) # (THR_M,THR_K,k)
    threadtileB = local_partition(blocktileB, threadlayoutB, threadIdx().x) # (THR_N,THR_K,k)

    threadtile_a = local_partition(a, threadlayoutA, threadIdx().x) # (THR_M,THR_K）
    threadtile_b = local_partition(b, threadlayoutB, threadIdx().x) # (THR_N,THR_K）

    tCsA = local_partition(a, threadlayoutC, threadIdx().x, (One(), :))  # (16,8)
    tCsB = local_partition(b, threadlayoutC, threadIdx().x)                  # (16,8)
    tCgC = local_partition(blocktileC, threadlayoutC, threadIdx().x)         # (16,16)

    frag_c = make_fragment_like(tCgC)

    k_max = size(threadtile_a, 2)

    for k in 1:k_max
        # copy gmem to smem
        cucopyto!(threadtile_a, view(threadtileA, (:, :, k)))
        cucopyto!(threadtile_b, view(threadtileB, (:, :, k)))
        sync_threads()
        gemm(tCsA, tCsB, frag_c)
        sync_threads()
    end
end

function gemm(A,B,C)
    M = size(A, 1)
    N = size(B, 1)
    K = size(A, 2)

    bM = static(128)
    bN = static(128)
    bK = static(8)

    blocklayoutA = make_layout((bM, bK))
    blocklayoutB = make_layout((bN, bK))
    blocklayoutC = make_layout((bM, bN))

    threadlayoutA = make_layout((static(32), static(8)))
    threadlayoutB = make_layout((static(32), static(8)))
    threadlayoutC = make_layout((static(16), static(16)))

    threads = dynamic(size(threadlayoutC))
    blocks = (cld(M, bM), cld(N, bN))

    @cuda threads=threads blocks=blocks gemme_kernel(M, N, K,
                                                      A, strideA, blocklayoutA, threadlayoutA,
                                                      B, strideB, blocklayoutB, threadlayoutB,
                                                      C, strideC, blocklayoutC, threadlayoutC)
end
