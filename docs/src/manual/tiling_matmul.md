# MatMul

This tutorial demonstrates how to perform matrix multiplication `C = A * B^T` using `MoYe`.

We follow the convention where the shape of matrix `A` is `(M, K)`, matrix `B` is `(N, K)`, and matrix `C` is `(M, N)`.

## On CPU

First, we launch Julia with 16 threads:

```julia
julia --threads 16
```

Correspondingly, we use a thread layout of

```julia
threadlayout = @Layout (4, 4)
```

We start with a simple example, where `M = 4, N = 4`, and `K = 8`. In this case, each thread processes one element of matrix `C`.

```julia
M = 4
N = 4
K = 8
A = reshape([i for i in 1:(M*K)], (M, K))
B = reshape([i for i in 1:(N*K)], (N, K))
C = zeros(Int, M, N)

moye_A = MoYeArray(A, (M, K))
moye_B = MoYeArray(B, (N, K))
moye_C = MoYeArray(C, (M, N))
```
After setting up the initial matrices and thread layouts, we can proceed with the tiling process. The [`@parallelize`](@ref) macro is used with a fourth argument, which is the projection. Specifically, the thread IDs of all columns in the thread layout are projected onto the first column, and only the first column is used for tiling. The static(1) is just a placeholder, representing the dimensions that are preserved in the projection.

```julia
threadlayout = @Layout (4, 4)
tile_A = @parallelize moye_A threadlayout 1 (static(1), :)
tile_B = @parallelize moye_B threadlayout 1 (:, static(1))
```

Let's take a look at the contents of tile_A:
```julia
julia> tile_A = @parallelize moye_A threadlayout 1 (static(1), :)
1×8 MoYeArray{Int64, 2, ViewEngine{Int64, Ptr{Int64}}, Layout{2, Tuple{Int64, Int64}, Tuple{StaticInt{4}, Int64}}}:
 1  5  9  13  17  21  25  29
```
Thread 1 processes the element in the first row and first column of `C`, and it needs the elements in the first row of `A`, which are the contents printed above. Let's see all the elements of `A` that Thread 2 needs:

```julia
julia> tile_A = @parallelize moye_A threadlayout 2 (static(1), :)
1×8 MoYeArray{Int64, 2, ViewEngine{Int64, Ptr{Int64}}, Layout{2, Tuple{Int64, Int64}, Tuple{StaticInt{4}, Int64}}}:
 2  6  10  14  18  22  26  30
```

This is exactly the second row of `A`. What about Thread 5? It is processing the entry at the first row and second column of `C`, so it also needs the first row of `A`.

```julia
julia> tile_A = @parallelize moye_A threadlayout 5 (static(1), :)
1×8 MoYeArray{Int64, 2, ViewEngine{Int64, Ptr{Int64}}, Layout{2, Tuple{Int64, Int64}, Tuple{StaticInt{4}, Int64}}}:
 1  5  9  13  17  21  25  29
```
Great, this is exactly what we want. Now let's look at `tile_B`. Threads 1 and 2 both need the first column of `B^T`, while Thread 5 needs the second column of `B^T`.
```julia
julia> tile_B = @parallelize moye

julia> tile_B = @parallelize moye_B threadlayout 1 (:, static(1))
1×8 MoYeArray{Int64, 2, ViewEngine{Int64, Ptr{Int64}}, Layout{2, Tuple{Int64, Int64}, Tuple{StaticInt{4}, Int64}}}:
 1  5  9  13  17  21  25  29

julia> tile_B = @parallelize moye_B threadlayout 2 (:, static(1))
1×8 MoYeArray{Int64, 2, ViewEngine{Int64, Ptr{Int64}}, Layout{2, Tuple{Int64, Int64}, Tuple{StaticInt{4}, Int64}}}:
 1  5  9  13  17  21  25  29

julia> tile_B = @parallelize moye_B threadlayout 5 (:, static(1))
1×8 MoYeArray{Int64, 2, ViewEngine{Int64, Ptr{Int64}}, Layout{2, Tuple{Int64, Int64}, Tuple{StaticInt{4}, Int64}}}:
 2  6  10  14  18  22  26  30
```

Once the tiling is done, we perform the matrix multiplication using three native loops. The outer loop iterates over the reduction mode `K`, while the inner two loops handle the row and column indices of the resulting matrix C. These loops are responsible for the actual computation of matrix C's elements by multiplying the corresponding elements of matrices A and B.

```julia
M = 4*32
N = 4*32
K = 8*32
A = rand(M, K)
B = rand(N, K)
C = zeros(M, N)

threadlayout = @Layout (4, 4)

moye_A = MoYeArray(A, (M, K))
moye_B = MoYeArray(B, (N, K))
moye_C = MoYeArray(C, (M, N))

Threads.@threads :static for i in 1:Threads.nthreads()
    tile_A = @parallelize moye_A threadlayout Threads.threadid() (static(1), :)
    tile_B = @parallelize moye_B threadlayout Threads.threadid() (:, static(1))
    tile_C = @parallelize moye_C threadlayout Threads.threadid()

    for k in axes(tile_A, 2)
        for n in axes(tile_C, 2)
            for m in axes(tile_C, 1)
                tile_C[m, n] += tile_A[m, k] * tile_B[n, k]
            end
        end
    end
end

C ≈ A * transpose(B)
```


## On GPU

```julia
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

    # Tiles for loading data A and B from gmem to smem
    threadtile_gA = @parallelize blocktile_A threadlayout_A threadIdx().x # (tM,tK,k)
    threadtile_sA = @parallelize sA threadlayout_A threadIdx().x # (tM,tK）

    threadtile_gB = @parallelize blocktile_B threadlayout_B threadIdx().x # (tN,tK,k)
    threadtile_sB = @parallelize sB threadlayout_B threadIdx().x # (tN,tK）

    # For mma computation
    computetile_sA = @parallelize sA threadlayout_C threadIdx().x (X, :)
    computetile_sB = @parallelize sB threadlayout_C threadIdx().x (:, X)
    computetile_gC = @parallelize blocktile_C threadlayout_C threadIdx().x 

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
    # copy rmem to gmem
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

    @cuda threads=threads blocks=blocks matmul_kernel(A, blocklayout_A, threadlayout_A,
                                                      B, blocklayout_B, threadlayout_B,
                                                      C, blocklayout_C, threadlayout_C)
end

function test()
    A = CUDA.randn(Float32, 2048, 256)
    B = CUDA.randn(Float32, 2048, 256)
    C = CUDA.randn(Float32, 2048, 2048)
    matmul(A, B, C)
    CUDA.synchronize()
    @test C == A * B'
    CUDA.unsafe_free!(A)
    CUDA.unsafe_free!(B)
    CUDA.unsafe_free!(C)
end
```
