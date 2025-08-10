# Tiled Matmul

While `@tile` and `@parallelize` are powerful tools for data manipulation, they can be cumbersome. `TiledCopy` and `TiledMMA` simplify this process.

## Tiled Copy

`TiledCopy` streamlines data transfer between arrays. Consider an example where six threads copy a 4x9 `src` array to a `dst` array of the same shape. The mapping of logical coordinates to thread IDs is as follows:

```
1 1 1 2 2 2 3 3 3
1 1 1 2 2 2 3 3 3
4 4 4 5 5 5 6 6 6
4 4 4 5 5 5 6 6 6
```

Each thread is assigned a data segment defined by `val_layout` (2,3):(1,2), while the thread group operates within `thr_layout` (2,3):(3,1).

First, initialize the arrays:

```@repl tiled_copy
using MoYe
src_buffer = collect(1:36) .* 0.1;
src = MoYeArray(src_buffer, @Layout((4,9)))
dst_buffer = zeros(36);
dst = MoYeArray(dst_buffer, make_layout((_4,_9)));
```

Next, set up the `TiledCopy`:
```@repl tiled_copy
thr_layout = @Layout (2, 3) (3, 1)
val_layout = @Layout (2, 3) (1, 2)
tiled_copy = make_tiled_copy(
	CopyAtom{UniversalCopy{Float64}, Float64}(),
	thr_layout, 
	val_layout)
```

The `Float64` in `CopyAtom` specifies the data type. `UniversalCopy{Float64}` indicates a non-vectorized copy. For vectorized copies, use a type like `UInt128`:
```julia
tiled_copy_vec = make_tiled_copy(
	CopyAtom{UniversalCopy{UInt128}, Float64}(),
	thr_layout, 
	val_layout)
```
Note that for vectorized copies, `val_layout` must have a divisible number of elements.

Visualize the `tiled_copy` using `print_typst(tiled_copy)` in the [Typst](https://typst.app) web app:

![matmuil](../assets/tiled_copy.svg)

The two tables show the thread distribution for `src` and `dst`. PTX instructions may reallocate each thread's data. For example:
```julia
print_typst(make_tiled_copy(MoYe.CopyAtom{LDSM_U32x4_N, UInt16}(),
                                          @Layout((16,2)), @Layout((2,4))));

```

![matmuil](../assets/ldmatrix.svg)

As shown, `thr_layout` and `val_layout` are defined on `dst`. We will revisit `ldmatrix` when discussing Tensor Cores.

After creating the `tiled_copy`, partition the data:

```@repl tiled_copy
thr_idx = 2;
thr_copy = get_slice(tiled_copy, thr_idx);
dst_t = partition_D(thr_copy, dst);
dst_t.layout
src_t = partition_S(thr_copy, src);
src_t.layout
copyto!(tiled_copy, dst_t, src_t);
dst
```

The second thread has now completed its copy. The shape of `dst_t` is `(CPY, CPY_M, CPY_K)`, where `CPY` is the number of vectorized values per thread. In this case, it's 1. Changing to `UniversalCopy{UInt128}` would alter this.

The NVIDIA Ampere architecture supports `cuda::memcpy_async` for asynchronous data copies between global and shared memory. In older architectures, this required intermediate registers:
```@repl tiled_copy
thr_idx = 3;
thr_copy = get_slice(tiled_copy, thr_idx);
dst_t = partition_D(thr_copy, dst);
src_t = partition_S(thr_copy, src);

dst_r = make_fragment_like(dst_t);
copyto!(tiled_copy, dst_r, src_t);
copyto!(tiled_copy, dst_t, dst_r);
dst
```

## TiledMMA

`TiledMMA` simplifies MMA partitions. Invoke `make_tiled_mma` as follows:
```@repl tiled_copy
mma_C = make_tiled_mma(UniversalFMA{TA,TB, TC}(), # MMA operation
                       @Layout((16,16)))          # Atom layout

```

You can replace `UniversalFMA` with other `MMAOp` types. View the predefined `MMAOps` with:
```@repl tiled_copy
MoYe.mma_ops_list
```

```julia
thr_mma = get_slice(mma_C, threadIdx().x);
tCsA = partition_A(sA);
tCsB = partition_B(sB);
tCgC = partition_C(gC);

tCrC = make_fragment_like(tCgC)
```
These instructions operate on Tensor Cores, which are covered in a later section.

## Matmul with Tiled Operations

Now, let's upgrade the `matmul_kernel` with `TiledCopy` and `TiledMMA`.

```julia
function matmul_kernel(A, sA_layout, copy_A,
                       B, sB_layout, copy_B,
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

    # Copy partition
    thr_copy_a = get_slice(copy_A, threadIdx().x)      
    tAgA = partition_S(thr_copy_a, gA)                 # (CPY, CPY_M, CPY_K, k)
    tAsA = partition_D(thr_copy_a, sA)                 # (CPY, CPY_M, CPY_K)
    tArA = make_fragment_like(tAsA)                    # (CPY, CPY_M, CPY_K)

    thr_copy_b = get_slice(copy_B, threadIdx().x)
    tBgB = partition_S(thr_copy_b, gB)                 # (CPY, CPY_N, CPY_K, k)
    tBsB = partition_D(thr_copy_b, sB)                 # (CPY, CPY_N, CPY_K)
    tBrB = make_fragment_like(tBsB)                    # (CPY, CPY_N, CPY_K)

    # MMA partition
    thr_mma = get_slice(mma_C, threadIdx().x)
    tCsA = partition_A(thr_mma, sA)                    # (MMA, MMA_M, MMA_K)
    tCsB = partition_B(thr_mma, sB)                    # (MMA, MMA_M, MMA_K)
    tCgC = partition_C(thr_mma, gC)                    # (MMA, MMA_M, MMA_N)

    # Overlap copy and compute
    copyto!(copy_A, tArA, view(tAgA, :, :, :, _1))
    copyto!(copy_B, tBrB, view(tBgB, :, :, :, _1))

    # Accumulator
    tCrC = make_fragment_C(thr_mma, tCgC)
    zeros!(tCrC)

    k_max = size(tAgA, 4)
    for k in 1:k_max
        sync_threads()
        copyto!(tAsA, tArA)
        copyto!(tBsB, tBrB)
        sync_threads()

	    # Load the next tile
	    k_next = k < k_max ? k+1 : k
	    copyto!(copy_A, tArA, view(tAgA, :, :, :, k_next))
	    copyto!(copy_B, tBrB, view(tBgB, :, :, :, k_next))

        @gc_preserve gemm!(mma_C, tCrC, tCsA, tCsB, tCrC)
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

    TA = eltype(A)
    TB = eltype(B)
    TC = eltype(C)
	
    copy_A = make_tiled_copy(CopyAtom{UniversalCopy{TA}, TA}(),
                             @Layout((32, 8)),
                             @Layout((1, 1)))
    copy_B = make_tiled_copy(CopyAtom{UniversalCopy{TB}, TB}(),
                             @Layout((32, 8)),
                             @Layout((1, 1)))

    mma_C = make_tiled_mma(UniversalFMA{TA,TB, TC}(), # MMA operation
                           @Layout((32,8)))          # Atom layout

    threads = Int(size(mma_C))
    blocks = (cld(size(A, 1), bM), cld(size(B, 1), bN))

    @cuda threads=threads blocks=blocks matmul_kernel(A, sA_layout, copy_A,
                                                      B, sB_layout, copy_B,
                                                      C, mma_C)
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
```

## Vectorized Copy and Memory Coalescing

As mentioned, you can use `UniversalCopy{Float64}` or `UniversalCopy{UInt128}` for vectorized copies. However, it is crucial to ensure that memory accesses are **coalesced**.

An uncoalesced copy:
```julia
copy_A = make_tiled_copy(CopyAtom{UniversalCopy{Float64}, TA}(),
                             @Layout((32, 8)),
                             @Layout((4, 1)))
```
Here, thread 1 loads from `[1], [2]` and thread 2 loads from `[5], [6]`, which is not coalesced.

Coalesced copies:
```julia
copy_A = make_tiled_copy(CopyAtom{UniversalCopy{Float64}, TA}(),
                             @Layout((32, 8)),
                             @Layout((2, 1)))
copy_A = make_tiled_copy(CopyAtom{UniversalCopy{UInt128}, TA}(),
                             @Layout((32, 8)),
                             @Layout((4, 1)))          
```
In these examples, threads access contiguous memory locations, leading to coalesced memory access and better performance.
