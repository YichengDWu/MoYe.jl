
using Test, MoYe, CUDA

if CUDA.functional() && MoYe.LLVM.version().major>=15
    @testset "Compile to LLVM" begin
        function kernel(op)
            A = CuStaticSharedArray(UInt16, (64,))
            a_frag = load(op, pointer(A))
            @cushow Float32(sum(a_frag))
            return nothing
        end

        op_to_intrinsic = Dict(MoYe.get_ldmatrix_ops())
        for op in Main.subtypes(MoYe.LdMatrix)
            buf = IOBuffer()
            @device_code_llvm io = buf @cuda threads=1 kernel(op())
            asm = String(take!(copy(buf)))

            @test occursin(op_to_intrinsic["$op"], asm)
        end
    end

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
          #  cucopyto!(threadtile_smem_A, threadtile_A)
         #   cucopyto!(threadtile_smem_B, threadtile_B)
         #   cp_async_wait()

            smem_B′ = MoYe.transpose(smem_B) # (N, K) (8, 16)

            frag_A = MoYeArray{Float16}(undef, @Layout((8,)))
            frag_B = MoYeArray{Float16}(undef, @Layout((4,)))
            frag_C = MoYeArray{Float16}(undef, @Layout((4,)))

            zeros!(frag_C)

            # loading from shared memory to registers
            ld_A = MoYe.LDSM_U32x4_N()
            ld_B = MoYe.LDSM_U32x2_N()

            # load A from shared memory
            recasted_smem_A = recast(UInt128, smem_A) # 16x2
            recasted_smem_B = recast(UInt128, smem_B′) # 8x2

            ptr_A = pointer(recasted_smem_A, Int(threadIdx().x))
            ptr_B = pointer(recasted_smem_B, mod1(Int(threadIdx().x), 16))

            recasted_ptr_A = recast(UInt32, ptr_A)
            recasted_ptr_B = recast(UInt32, ptr_B)

            llvmstruct_A = ld_A(recasted_ptr_A) # 4 UInt32
            llvmstruct_B = ld_B(recasted_ptr_B) # 2 UInt32

            recasted_frag_A = recast(UInt32, frag_A)
            recasted_frag_B = recast(UInt32, frag_B)

            recasted_frag_A[1] = getfield(llvmstruct_A, 1)
            recasted_frag_A[2] = getfield(llvmstruct_A, 2)
            recasted_frag_A[3] = getfield(llvmstruct_A, 3)
            recasted_frag_A[4] = getfield(llvmstruct_A, 4)

            recasted_frag_B[1] = getfield(llvmstruct_B, 1)
            recasted_frag_B[2] = getfield(llvmstruct_B, 2)

            # good syntax here
            traits = MoYe.MMATraits{MoYe.MMAOP_16x8x16_F16F16F16F16_TN}()
            @gc_preserve MoYe.mma_unpack!(traits, frag_C, frag_A, frag_B, frag_C)

            recasted_moye_C = recast(UInt32, moye_C) # 16x4
            recasted_frag_C = recast(UInt32, frag_C) # 2x1

            row, col = fldmod1(Int(threadIdx().x), 4)

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
