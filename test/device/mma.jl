using Test, CuTe, CUDA

@inline CuTe.Registers{T,S}() where {T, S} = ArrayEngine{T}(undef, static(S))

function kernel()
    op = MMA_16x8x16_F16F16F16F16_TN()
    a_frag = op.ARegisters()
    b_frag = op.BRegisters()
    c_frag = op.CRegisters()

    d_frag = mma(op, a_frag, b_frag, c_frag) # this should be a ArrayEngine in the future
    return
end

@test_nowarn @cuda threads=32 kernel()
