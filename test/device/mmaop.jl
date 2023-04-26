using Test, Shambles, CUDA

if CUDA.functional()
    @inline Shambles.Registers{T,S}() where {T, S} = ArrayEngine{T}(undef, static(S))
    @inline _tofloat32(x::VecElement) = convert(Float32, x.value)
    @inline _tofloat32(x::Number) = convert(Float32, x)

    @testset "Compile to LLVM" begin
        function kernel(op)
            a_frag = op.ARegisters()
            b_frag = op.BRegisters()
            c_frag = op.CRegisters()

            d_frag = mma(a_frag, b_frag, c_frag, op) # this should be a ArrayEngine in the future
            @cushow _tofloat32(d_frag[1][1])
            return
        end

        op_to_intrinsic = Dict(Shambles.get_mma_ops())
        for op in Base.subtypes(MMAOP)
            buf = IOBuffer()
            @device_code_llvm io = buf @cuda threads=32 kernel(op())
            asm = String(take!(copy(buf)))

            @test occursin(op_to_intrinsic["$op"], asm)
        end
    end
end
