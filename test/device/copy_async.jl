using Shambles, Test, CUDA

function copy_kernel(M, N, dest, src, blocklayout, threadlayout)
    smem = Shambles.SharedMemory(eltype(dest), cosize(blocklayout))
    cute_smem = CuTeArray(smem, blocklayout)


    cute_dest = CuTeArray(pointer(dest), make_layout((256, 32))) # bug: cannot use (M, N)
    cute_src = CuTeArray(pointer(src), make_layout((256, 32)))

    bM = size(blocklayout, 1)
    bN = size(blocklayout, 2)

    blocktile_dest = local_tile(cute_dest, (bM, bN), (Int(blockIdx().x), Int(blockIdx().y)))
    blocktile_src = local_tile(cute_src, (bM, bN), (Int(blockIdx().x), Int(blockIdx().y)))

    threadtile_dest = local_partition(blocktile_dest, threadlayout, Int(threadIdx().x))
    threadtile_src = local_partition(blocktile_src, threadlayout, Int(threadIdx().x))
    threadtile_smem = local_partition(cute_smem, threadlayout, Int(threadIdx().x))

   #cucopyto!(threadtile_smem, threadtile_src) # bug
    sync_threads()
   # cucopyto!(threadtile_dest, threadtile_smem)
  #  sync_threads()
    return nothing
end

M = 256
N = 32

a = CUDA.rand(Float32, M, N)
b = CUDA.rand(Float32, M, N)

blocklayout = @Layout((128, 16))
threadlayout = @Layout((32, 8))

bM = size(blocklayout, 1)
bN = size(blocklayout, 2)

blocks = (cld(M, bM), cld(N, bN))
threads = Shambles.Static.dynamic(size(threadlayout))

@cuda blocks=blocks threads=threads copy_kernel(M,N,a, b, blocklayout, threadlayout)
