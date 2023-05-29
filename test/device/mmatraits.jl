using Test, MoYe, CUDA

@testset "Size of MMATraits" begin
    for mmaop in [MoYe.MMAOP_16x8x8_F16F16F16F16_TN,
        MoYe.MMAOP_16x8x8_F32F16F16F32_TN,
        MoYe.MMAOP_16x8x8_F32BF16BF16F32_TN,
        MoYe.MMAOP_16x8x16_F16F16F16F16_TN,
        MoYe.MMAOP_16x8x16_F32F16F16F32_TN,
        MoYe.MMAOP_16x8x16_F32BF16BF16F32_TN]
        @test sizeof(MoYe.MMATraits{mmaop}()) == 0
    end
end

if CUDA.functional()
    @testset "16x8x16_F16F16F16F16" begin
    function kernel(A,B,C, smemlayout_A, smemlayout_B, thread_layout)
        moye_A = MoYeArray(pointer(A), @Layout((16,16))) # M-major
        moye_B = MoYeArray(pointer(B), @Layout((16,8)))  # K-major
        moye_C = MoYeArray(pointer(C.parent), @Layout((16, 8), (8, 1)))

        smem_A = MoYeSharedArray(Float16, smemlayout_A) # K-major
        smem_B = MoYeSharedArray(Float16, smemlayout_B) # K-major

        threadtile_A = @parallelize moye_A thread_layout threadIdx().x
        threadtile_B = @parallelize moye_B thread_layout threadIdx().x

        threadtile_smem_A = @parallelize smem_A thread_layout threadIdx().x
        threadtile_smem_B = @parallelize smem_B thread_layout threadIdx().x

        for i in eachindex(threadtile_A)
            threadtile_smem_A[i] = threadtile_A[i]
        end

        for i in eachindex(threadtile_B)
            threadtile_smem_B[i] = threadtile_B[i]
        end

        sync_threads()

        smem_B′ = MoYe.transpose(smem_B) # (N, K) (8, 16)

        frag_A = MoYeArray{Float16}(undef, @Layout((8,)))
        frag_B = MoYeArray{Float16}(undef, @Layout((4,)))
        frag_C = MoYeArray{Float16}(undef, @Layout((4,)))

        zeros!(frag_C)

        # loading from shared memory to registers
        ld_A = MoYe.LDSM_U32x4_N()
        ld_B = MoYe.LDSM_U32x2_N()

        recasted_smem_A = recast(UInt128, smem_A) # 16x2
        recasted_smem_B = recast(UInt128, smem_B′) # 8x2

        # or parallelize then recast
        copytile_smem_A = @parallelize recasted_smem_A @Layout((16, 2)) threadIdx().x
        copytile_smem_B = @parallelize recasted_smem_B @Layout((8, 2)) threadIdx().x

        recasted_frag_A = recast(UInt32, frag_A)
        recasted_frag_B = recast(UInt32, frag_B)

        copyto!(ld_A, recasted_frag_A, copytile_smem_A)
        copyto!(ld_B, recasted_frag_B, copytile_smem_B)

        # good syntax here
        traits = MoYe.MMATraits{MoYe.MMAOP_16x8x16_F16F16F16F16_TN}()
        @gc_preserve MoYe.mma_unpack!(traits, frag_C, frag_A, frag_B, frag_C)

        recasted_moye_C = recast(UInt32, moye_C) # 16x4
        recasted_frag_C = recast(UInt32, frag_C) # 2x1

        row, col = fldmod1(Int(threadIdx().x), 4)

        # awkward manual indexing
        recasted_moye_C[row, col] = recasted_frag_C[1]
        recasted_moye_C[row+8, col] = recasted_frag_C[2]
        return nothing
    end

    smemlayout_A = @Layout((16, 16), (16, 1))
    smemlayout_B = @Layout((16,  8), (1, 16))
    thread_layout = @Layout (16, 2)

    A = CUDA.rand(Float16, 16, 16)
    B = CUDA.rand(Float16, 16, 8)
    C = transpose(CUDA.rand(Float16, 8, 16)) # row-major, this is awkward
    @cuda threads=32 kernel(A,B,C, smemlayout_A, smemlayout_B, thread_layout)
    CUDA.synchronize()
    @test A * B ≈ C
    end
end
