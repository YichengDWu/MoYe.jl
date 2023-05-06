# Tiling MatMul

This tutorial demonstrates how to perform matrix multiplication `C = A * B^T + C` using tiling techniques.

We follow the convention where the shape of matrix `A` is `(M, K)`, matrix `B` is `(N, K)`, and matrix `C` is `(M, N)`.

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

moye_A = MoYeArray(pointer(A), (M, K))
moye_B = MoYeArray(pointer(B), (N, K))
moye_C = MoYeArray(pointer(C), (M, N))
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

moye_A = MoYeArray(pointer(A), (M, K))
moye_B = MoYeArray(pointer(B), (N, K))
moye_C = MoYeArray(pointer(C), (M, N))

tile_A = @parallelize moye_A threadlayout 1 (static(1), :)
tile_B = @parallelize moye_B threadlayout 1 (:, static(1))

moye_A = MoYeArray(pointer(A), (M, K))
moye_B = MoYeArray(pointer(B), (N, K))
moye_C = MoYeArray(pointer(C), (M, N))

GC.@preserve A B C begin
    Threads.@threads :static for i in 1:Threads.nthreads()
        tile_A = @parallelize moye_A threadlayout Threads.threadid() (static(1), :)
        tile_B = @parallelize moye_B threadlayout Threads.threadid() (:, static(1))
        tile_C = @parallelize moye_C threadlayout Threads.threadid()

        for k in axes(tile_A, 2)
            for m in axes(tile_C, 1)
                for n in axes(tile_C, 2)
                    tile_C[m, n] += tile_A[m, k] * tile_B[n, k]
                end
            end
        end
    end
end

C ≈ A * transpose(B)
```
