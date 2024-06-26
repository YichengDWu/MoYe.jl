using MoYe, Test
# C = A * B^T

M = 4*32
N = 4*32
K = 8*32
A = rand(M, K)
B = rand(N, K)
C = zeros(M, N)

threadlayout = @Layout (4, 4)

moye_A = MoYeArray(pointer(A), (M, K))
moye_B = MoYeArray(pointer(B), (N, K))
moye_C = MoYeArray(pointer(C), (M, N))

tile_A = @parallelize moye_A threadlayout 1 (_1, :)
tile_B = @parallelize moye_B threadlayout 1 (:, _1)

GC.@preserve A B C begin
    moye_A = MoYeArray(pointer(A), (M, K))
    moye_B = MoYeArray(pointer(B), (N, K))
    moye_C = MoYeArray(pointer(C), (M, N))

    Threads.@threads :static for i in 1:Threads.nthreads()

        tile_A = @parallelize moye_A threadlayout Threads.threadid() (_1, :)
        tile_B = @parallelize moye_B threadlayout Threads.threadid() (:, _1)
        tile_C = @parallelize moye_C threadlayout Threads.threadid()

        for k in 1:size(tile_A, 2)
            for m in 1:size(tile_C, 1)
                for n in 1:size(tile_C, 2)
                    tile_C[m, n] += tile_A[m, k] * tile_B[n, k]
                end
            end
        end
    end
end

@test C ≈ A * transpose(B)
