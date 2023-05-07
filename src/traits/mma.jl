abstract type AbstractTraits end
abstract type AbstractMMATraits{M, DElType, AElType, BElType, CElType} <: AbstractTraits end


# in the future, we might have `HopperMMATraits` and so on.
struct MMATraits{OP <: AbstractMMAOP, DElType, AElType, BElType, CElType, S, T, A, B, C} <: AbstractMMATraits{OP, DElType, AElType, BElType, CElType}
    mma_op::OP
    mnk::S
    threadid::T
    Alayout::A
    Blayout::B
    Clayout::C
end

function MMATraits{M, DElType, AElType, BElType, CElType}(mnk, threadid, Alayout, Blayout,
                                                          Clayout) where {M <: AbstractMMAOP,
                                                                          DElType, AElType,
                                                                          BElType, CElType}
    return MMATraits{M, DElType, AElType, BElType, CElType, typeof(mnk), typeof(threadid),
                     typeof(Alayout), typeof(Blayout), typeof(Clayout)}(M(), mnk, threadid,
                                                                        Alayout, Blayout,
                                                                        Clayout)
end

function MMATraits{M}(mnk, threadid, Alayout, Blayout, Clayout) where {M <: AbstractMMAOP}
    mma_op = M()
    DElType = eltype(mma_op.DRegisters)
    AElType = eltype(mma_op.ARegisters)
    BElType = eltype(mma_op.BRegisters)
    CElType = eltype(mma_op.CRegisters)

    return MMATraits{M, DElType, AElType, BElType, CElType, typeof(mnk), typeof(threadid),
                     typeof(Alayout), typeof(Blayout), typeof(Clayout)}(mma_op, mnk,
                                                                        threadid, Alayout,
                                                                        Blayout, Clayout)
end

for (fn, ElType) in Dict(:valtype_d => :DElType, :valtype_a => :AElType,
                         :valtype_b => :BElType, :valtype_c => :CElType)
    @eval $fn(::AbstractMMATraits{M, DElType, AElType, BElType, CElType}) where {M,
                                                                         DElType,
                                                                         AElType,
                                                                         BElType,
                                                                         CElType} =
        $ElType
end

# default implementation, Hooper would need to specialize on these
frgtype_d(traits::AbstractMMATraits) = valtype_d(traits)
frgtype_a(traits::AbstractMMATraits) = valtype_a(traits)
frgtype_b(traits::AbstractMMATraits) = valtype_b(traits)
frgtype_c(traits::AbstractMMATraits) = valtype_c(traits)

function MMATraits{UniversalFMA{D, A, B, C}}() where {D, A, B, C}
    mnk = (static(1), static(1), static(1))
    threadid = @Layout 1
    Alayout = @Layout (1, 1)
    Blayout = @Layout (1, 1)
    Clayout = @Layout (1, 1)
    return MMATraits{UniversalFMA{D, A, B, C}, D, A, B, C}(mnk, threadid, Alayout, Blayout,
                                                           Clayout)
end

# used for dispatching
const LocalArray{T, N, L} = MoYeArray{T, N, ViewEngine{T, Ptr{T}}, L}
const SharedArray{T, N, L} = MoYeArray{T, N, ViewEngine{T, LLVMPtr{T, AS.Shared}}, L}

# again, default implementation, Hooper would need to specialize on it
function mma_unpack!(traits::AbstractMMATraits{M, TD, TA, TB, TC},
                     D::LocalArray{TD}, A::LocalArray{TA},
                     B::LocalArray{TB}, C::LocalArray{TC}) where {M, TD, TA, TB, TC}
    RegTypeD = regtype_d(traits.mma_op)
    RegTypeA = regtype_a(traits.mma_op)
    RegTypeB = regtype_b(traits.mma_op)
    RegTypeC = regtype_c(traits.mma_op)

    RegNumD = regnum_d(traits.mma_op)
    RegNumA = regnum_a(traits.mma_op)
    RegNumB = regnum_b(traits.mma_op)
    RegNumC = regnum_c(traits.mma_op)

    rD = recast(RegTypeD, D)
    rA = recast(RegTypeA, A)
    rB = recast(RegTypeB, B)
    rC = recast(RegTypeC, C)

    @assert length(rD) == RegNumD
    @assert length(rA) == RegNumA
    @assert length(rB) == RegNumB
    @assert length(rC) == RegNumC
    fma!(traits.mma_op, rD, rA, rB, rC)
end
