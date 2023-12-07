using MoYe, Test, CUDA

function copy_kernel(dest, src, smemlayout, blocklayout, threadlayout)
    smem = MoYe.SharedMemory(eltype(dest), cosize(smemlayout))
    moye_smem = MoYeArray(smem, smemlayout)

    moye_dest = MoYeArray(dest)
    moye_src = MoYeArray(src)

    bM = size(blocklayout, 1)
    bN = size(blocklayout, 2)

    blocktile_dest = @tile moye_dest (bM, bN) (blockIdx().x, blockIdx().y)
    blocktile_src  = @tile moye_src  (bM, bN) (blockIdx().x, blockIdx().y)

    threadtile_dest = @parallelize blocktile_dest threadlayout threadIdx().x
    threadtile_src  = @parallelize blocktile_src  threadlayout threadIdx().x
    threadtile_smem = @parallelize moye_smem      threadlayout threadIdx().x

    copyto!(threadtile_smem, threadtile_src)
    cp_async_wait()
    copyto!(threadtile_dest, threadtile_smem)

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

    @cuda blocks=blocks threads=threads copy_kernel(a, b, smemlayout, blocklayout, threadlayout)
    CUDA.synchronize()
    @test a == b
end

if CUDA.functional()
    test_copy_async(2048, 2048)
end


function transpose_kernel(dest, src, smemlayout, blocklayout, threadlayout)
    moye_smem = MoYeSharedArray(eltype(dest), smemlayout)

    moye_src = MoYeArray(src)
    moye_dest = MoYeArray(dest)

    bM = size(blocklayout, 1)
    bN = size(blocklayout, 2)

    blocktile_src  = @tile moye_src  (bM, bN) (blockIdx().x, blockIdx().y)
    blocktile_dest = @tile moye_dest (bN, bM) (blockIdx().y, blockIdx().x)

    threadtile_dest = @parallelize blocktile_dest threadlayout threadIdx().x
    threadtile_src  = @parallelize blocktile_src  threadlayout threadIdx().x
    threadtile_smem = @parallelize moye_smem      threadlayout threadIdx().x

    copyto!(threadtile_smem, threadtile_src)
    cp_async_wait()
    sync_threads()

    moye_smem′ = MoYe.transpose(moye_smem)
    threadtile_smem′ = @parallelize moye_smem′ threadlayout threadIdx().x

    copyto!(threadtile_dest, threadtile_smem′)
    return nothing
end


function test_transpose(M, N)
    a = CUDA.rand(Float32, M, N)
    b = CUDA.rand(Float32, N, M)

    blocklayout = @Layout (32, 32)
    smemlayout = @Layout (32, 32) (1, 33)
    threadlayout = @Layout (32, 8)

    bM = size(blocklayout, 1)
    bN = size(blocklayout, 2)

    blocks = (cld(M, bM), cld(N, bN))
    threads = MoYe.dynamic(size(threadlayout))

    @cuda blocks=blocks threads=threads transpose_kernel(a, b, smemlayout, blocklayout, threadlayout)
    CUDA.synchronize()
    @test a == transpose(b)
end

if CUDA.functional()
    test_transpose(2048, 2048)
end
