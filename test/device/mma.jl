using Test, CuTe, CUDA

if CUDA.functional()
    @inline CuTe.Registers{T,S}() where {T, S} = ArrayEngine{T}(undef, static(S))

    function kernel(op)
        a_frag = op.ARegisters()
        b_frag = op.BRegisters()
        c_frag = op.CRegisters()

        d_frag = mma(a_frag, b_frag, c_frag, op) # this should be a ArrayEngine in the future

        s = d_frag[1][1].value
        @cushow Float32(s)
        return
    end

    buf = IOBuffer()
    @device_code_ptx io = buf @cuda threads=32 kernel(MMA_16x8x16_F16F16F16F16_TN())
    asm = String(take!(copy(buf)))
    @test occursin("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16", asm)

end
