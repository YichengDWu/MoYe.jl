using MoYe, Test, CUDA

function copy_kernel(M, N, dest, src, smemlayout, blocklayout, threadlayout)
    smem = MoYe.SharedMemory(eltype(dest), cosize(smemlayout))
    moye_smem = MoYeArray(smem, smemlayout)

    moye_dest = MoYeArray(pointer(dest), Layout((M, N), (static(1), M))) # bug: cannot use make_layout((M, N))
    moye_src = MoYeArray(pointer(src), Layout((M, N), (static(1), M)))

    bM = size(blocklayout, 1)
    bN = size(blocklayout, 2)

    blocktile_dest = @tile moye_dest (bM, bN) (blockIdx().x, blockIdx().y)
    blocktile_src  = @tile moye_src  (bM, bN) (blockIdx().x, blockIdx().y)

    threadtile_dest = @parallelize blocktile_dest threadlayout threadIdx().x
    threadtile_src  = @parallelize blocktile_src  threadlayout threadIdx().x
    threadtile_smem = @parallelize moye_smem      threadlayout threadIdx().x

    cucopyto!(threadtile_smem, threadtile_src)
    cp_async_wait()
    cucopyto!(threadtile_dest, threadtile_smem)

    return nothing
end

function test_copy_async(M, N)
    a = CUDA.rand(Float32, M, N)
    b = CUDA.rand(Float32, M, N)

    blocklayout = @Layout (32, 32) # 32 * 32 elements in a block
    smemlayout = @Layout (32, 32)  # 32 * 32 elements in shared memory
    threadlayout = @Layout (32, 8) # 32 * 8 threads in a block

    bM = size(blocklayout, 1)
    bN = size(blocklayout, 2)

    blocks = (cld(M, bM), cld(N, bN))
    threads = MoYe.dynamic(size(threadlayout))

    @cuda blocks=blocks threads=threads copy_kernel(M, N, a, b, smemlayout, blocklayout, threadlayout)
    CUDA.synchronize()
    @test a == b
end

test_copy_async(2048, 2048)

function bench_copy(a,b)
    M = size(a, 1)
    N = size(a, 2)
    blocklayout = @Layout (32, 32) # 32 * 32 elements in a block
    smemlayout = @Layout (32, 32)  # 32 * 32 elements in shared memory
    threadlayout = @Layout (32, 8) # 32 * 8 threads in a block

    bM = size(blocklayout, 1)
    bN = size(blocklayout, 2)

    blocks = (cld(M, bM), cld(N, bN))
    threads = MoYe.dynamic(size(threadlayout))

    @cuda blocks=blocks threads=threads copy_kernel(M, N, a, b, smemlayout, blocklayout, threadlayout)
end
