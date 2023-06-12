using CUDA, MoYe, Test
function tiled_copy_kernel(g_in, g_out, tiled_copy, smem_layout)
    t_g_in = MoYeArray(pointer(g_in), smem_layout)
    t_g_out = MoYeArray(pointer(g_out), smem_layout)
    t_smem=MoYeSharedArray(UInt16, smem_layout)
    tid=Int(threadIdx().x)

    for i in tid:size(tiled_copy):size(t_smem.layout)
        @inbounds  t_smem[i] = t_g_in[i]
    end

    thr_copy = get_thread_slice(tiled_copy, tid)
    tXsX = partition_S(thr_copy, t_smem)
    tXgX = partition_D(thr_copy, t_g_out)
    tXrX = MoYeArray{UInt16}(undef, tXgX.layout.shape)

    # smem to rmem
    copyto!(tiled_copy, tXrX, tXsX)
    # rmem to gmem
    copyto!(tXgX, tXrX)
    @inbounds tXrX.engine[1] # bug, have to load
    return nothing
end

@testset "UniversalCopy" begin
    @testset "32 x 32" begin
        g_in = [UInt16(i) for i in 1:32*32]
        g_out = zeros(UInt16, 32*32)
        cu_g_in = CuArray(g_in)
        cu_g_out = CuArray(g_out)

        smem_layout = @Layout (32,32) (1,32)
        tiled_copy = make_tiled_copy(MoYe.CopyAtom{MoYe.UniversalCopy{UInt16, UInt16}, UInt16}(),
                                    @Layout((16,2)), @Layout((2,4)))

        @cuda threads=32 tiled_copy_kernel(cu_g_in, cu_g_out, tiled_copy, smem_layout)
        @test cu_g_out == cu_g_in
    end

    @testset "32 x 8" begin
        g_in = [UInt16(i) for i in 1:32*8]
        g_out = zeros(UInt16, 32*8)
        cu_g_in = CuArray(g_in)
        cu_g_out = CuArray(g_out)

        smem_layout = @Layout (32, (2, 4)) (2, (1, 64))
        tiled_copy = make_tiled_copy(MoYe.CopyAtom{MoYe.UniversalCopy{UInt16, UInt16}, UInt16}(),
                                    @Layout((32,1)), @Layout((1,8)))
        @cuda threads=32 tiled_copy_kernel(cu_g_in, cu_g_out, tiled_copy, smem_layout)
        @test cu_g_out == cu_g_in
    end
end

@testset "LDMATRIX" begin
    @testset "32 x 32" begin
        g_in = [UInt16(i) for i in 1:32*32]
        g_out = zeros(UInt16, 32*32)
        smem_layout = @Layout (32,32) (1,32)
        cu_g_in = CuArray(g_in)
        cu_g_out = CuArray(g_out)
        for ldmatrix in [:LDSM_U32x1_N, :LDSM_U32x2_N, :LDSM_U32x4_N]
            @testset "$ldmatrix" begin
                @eval tiled_copy = make_tiled_copy(MoYe.CopyAtom{$ldmatrix, UInt16}(),
                                                  @Layout((16,2)), @Layout((2,4)))
                @cuda threads=32 tiled_copy_kernel(cu_g_in, cu_g_out, tiled_copy, smem_layout)
                @test cu_g_out == cu_g_in
                fill!(cu_g_out, zero(UInt16))
            end
        end
    end

    @testset "32 x 8" begin
        g_in = [UInt16(i) for i in 1:32*8]
        g_out = zeros(UInt16, 32*8)
        cu_g_in = CuArray(g_in)
        cu_g_out = CuArray(g_out)

        smem_layout = @Layout (32, (2, 4)) (2, (1, 64))
        for ldmatrix in [:LDSM_U32x1_N, :LDSM_U32x2_N, :LDSM_U32x4_N]
            @testset "$ldmatrix" begin
                @eval tiled_copy = make_tiled_copy(MoYe.CopyAtom{$ldmatrix, UInt16}(),
                                                  @Layout((32,1)), @Layout((1,8)))
                @cuda threads=32 tiled_copy_kernel(cu_g_in, cu_g_out, tiled_copy, smem_layout)
                @test cu_g_out == cu_g_in
                fill!(cu_g_out, zero(UInt16))
            end
        end
    end
end
