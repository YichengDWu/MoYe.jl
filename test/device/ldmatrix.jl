
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
end
