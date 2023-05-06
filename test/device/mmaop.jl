using Test, MoYe, CUDA

if CUDA.functional()
    @inline MoYe.Registers{T,S}() where {T, S} = ArrayEngine{T}(undef, static(S))
    @inline _tofloat32(x::VecElement) = convert(Float32, x.value)
    @inline _tofloat32(x::Number) = convert(Float32, x)

    @testset "Compile to LLVM" begin
        function kernel(mma_op)
            a_frag = mma_op.ARegisters()
            b_frag = mma_op.BRegisters()
            c_frag = mma_op.CRegisters()

            d_frag = mma_op(a_frag, b_frag, c_frag) # this should be a ArrayEngine in the future
            @cushow _tofloat32(d_frag[1][1])
            return
        end

        op_to_intrinsic = Dict(MoYe.get_mma_ops())
        for op in Main.subtypes(MoYe.MMAOP)
            buf = IOBuffer()
            @device_code_llvm io = buf @cuda threads=32 kernel(op())
            asm = String(take!(copy(buf)))

            @test occursin(op_to_intrinsic["$op"], asm)
        end
    end
end
