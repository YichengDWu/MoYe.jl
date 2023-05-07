using Test, MoYe, CUDA

if CUDA.functional()
    @inline make_fragment(::Type{MoYe.Registers{T,S}}) where {T, S} = MoYeArray{T}(undef, (static(S),))
    _float32(x::AbstractFloat) = Float32(x)
    _float32(x::VecElement) = Float32(x.value)
    _one(::Type{NTuple{N, VecElement{T}}}) where {N, T} = ntuple(i -> VecElement(one(T)), Val(N))
    _one(x) = one(x)

    @testset "Compile to LLVM" begin
        function kernel(mma_op)
            a_frag = make_fragment(mma_op.ARegisters)
            b_frag = make_fragment(mma_op.BRegisters)
            c_frag = make_fragment(mma_op.CRegisters)

            fill!(a_frag, _one(eltype(a_frag)))
            fill!(b_frag, _one(eltype(b_frag)))
            fill!(c_frag, _one(eltype(c_frag)))

            d_frag = mma_op(a_frag, b_frag, c_frag)
            @cushow _float32(getfield(d_frag,1)[1])
            return
        end

        op_to_intrinsic = Dict(MoYe.get_mma_ops())
        for op_name in keys(op_to_intrinsic)
            op = @eval MoYe.$(Symbol(op_name))
            buf = IOBuffer()
            @device_code_llvm io = buf @cuda threads=32 kernel(op())
            asm = String(take!(copy(buf)))
            @test occursin(get(op_to_intrinsic, "$op", ""), asm)
        end
    end
end
