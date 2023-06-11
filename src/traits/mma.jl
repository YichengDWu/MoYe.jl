abstract type AbstractTraits{OP} end
abstract type AbstractMMATraits{OP} <: AbstractTraits{OP} end

# in the future, we might have `HopperMMATraits` and so on.
struct MMATraits{OP <: AbstractMMAOP, DElType, AElType, BElType, CElType, S, T, A, B, C} <:
       AbstractMMATraits{OP}
    mma_op::OP
    MNK::S
    threadid::T
    Alayout::A
    Blayout::B
    Clayout::C
end

function MMATraits{M, DElType, AElType, BElType, CElType}(MNK, threadid, Alayout, Blayout,
                                                          Clayout) where {
                                                                          M <:
                                                                          AbstractMMAOP,
                                                                          DElType, AElType,
                                                                          BElType, CElType}
    return MMATraits{M, DElType, AElType, BElType, CElType, typeof(MNK), typeof(threadid),
                     typeof(Alayout), typeof(Blayout), typeof(Clayout)}(M(), MNK, threadid,
                                                                        Alayout, Blayout,
                                                                        Clayout)
end

function MMATraits{M}(MNK, threadid, Alayout, Blayout, Clayout) where {M <: AbstractMMAOP}
    mma_op = M()
    DElType = eltype(mma_op.DRegisters)
    AElType = eltype(mma_op.ARegisters)
    BElType = eltype(mma_op.BRegisters)
    CElType = eltype(mma_op.CRegisters)

    return MMATraits{M, DElType, AElType, BElType, CElType, typeof(MNK), typeof(threadid),
                     typeof(Alayout), typeof(Blayout), typeof(Clayout)}(mma_op, MNK,
                                                                        threadid, Alayout,
                                                                        Blayout, Clayout)
end

for (fn, ElType) in Dict(:valtype_d => :DElType, :valtype_a => :AElType,
                         :valtype_b => :BElType, :valtype_c => :CElType)
    @eval function $fn(::MMATraits{M, DElType, AElType, BElType, CElType}) where {M,
                                                                                  DElType,
                                                                                  AElType,
                                                                                  BElType,
                                                                                  CElType}
        return $ElType
    end
end

# default implementation, Hooper would need to specialize on these
frgtype_d(traits::AbstractMMATraits) = valtype_d(traits)
frgtype_a(traits::AbstractMMATraits) = valtype_a(traits)
frgtype_b(traits::AbstractMMATraits) = valtype_b(traits)
frgtype_c(traits::AbstractMMATraits) = valtype_c(traits)

function MMATraits{UniversalFMA{D, A, B, C}}() where {D, A, B, C}
    MNK = (static(1), static(1), static(1))
    threadid = @Layout 1
    Alayout = @Layout (1, 1)
    Blayout = @Layout (1, 1)
    Clayout = @Layout (1, 1)
    return MMATraits{UniversalFMA{D, A, B, C}, D, A, B, C}(MNK, threadid, Alayout, Blayout,
                                                           Clayout)
end

@inline mma_op(mma_traits::AbstractMMATraits) = mma_traits.mma_op

# again, default implementation, Hooper would need to specialize on it
function mma_unpack!(traits::MMATraits{OP, TD, TA, TB, TC}, D::LocalArray{TD},
                     A::LocalArray{TA}, B::LocalArray{TB},
                     C::LocalArray{TC}) where {OP, TD, TA, TB, TC}
    mma_op = mma_op(traits)
    RegTypeD = regtype_d(mma_op)
    RegTypeA = regtype_a(mma_op)
    RegTypeB = regtype_b(mma_op)
    RegTypeC = regtype_c(mma_op)

    RegNumD = regnum_d(mma_op)
    RegNumA = regnum_a(mma_op)
    RegNumB = regnum_b(mma_op)
    RegNumC = regnum_c(mma_op)

    rD = recast(RegTypeD, D)
    rA = recast(RegTypeA, A)
    rB = recast(RegTypeB, B)
    rC = recast(RegTypeC, C)

    @assert length(rD) == RegNumD
    @assert length(rA) == RegNumA
    @assert length(rB) == RegNumB
    @assert length(rC) == RegNumC
    return fma!(mma_op, rD, rA, rB, rC)
end
