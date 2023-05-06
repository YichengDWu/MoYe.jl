abstract type AbstractAtom{Traits} end
abstract type AbstractMMAAtom{Traits} <: AbstractAtom{Traits}end

function apply(mma_atom::AbstractMMAAtom, D::MoYeArray{TD,1}, A::MoYeArray{TA,1},
               B::MoYeArray{TB,1}, C::MoYeArray{TC,1}) where {TD, TA, TB, TC}
    return mma_unpack!(mma_atom, D, A, B, C)
end

function apply(mma_atom::AbstractMMAAtom, A::MoYeArray, B::MoYeArray, C::MoYeArray)
    return apply(mma_atom, C, A, B, C)
end

function make_fragment_C(m::AbstractMMAAtom, C::MoYeArray{T, N}) where {T,N}
    @assert rank(C) ≥ 3
    @assert size(C.layout, 1) == size(m.traits.Clayout, 2)
    return MoYeArray{frgtype_c(m.traits)}(undef, shape(C)) # (V, M, N)
end

# Hopper needs to specialize on make_fragment_A and make_fragment_B
function make_fragment_A(m::AbstractMMAAtom, A::MoYeArray{T, N}) where {T,N}
    @assert rank(A) ≥ 3
    @assert size(A.layout, 1) == size(m.traits.Alayout, 2)
    return MoYeArray{frgtype_a(m.traits)}(undef, shape(A)) # (V, M, K)
end

function make_fragment_B(m::AbstractMMAAtom, B::MoYeArray{T, N}) where {T,N}
    @assert rank(B) ≥ 3
    @assert size(B.layout, 1) == size(m.traits.Blayout, 2)
    return MoYeArray{frgtype_b(m.traits)}(undef, shape(B)) # (V, K, N)
end


struct MMAAtom{Traits <: MMATraits, ARGS} <: AbstractMMAAtom{Traits}
    traits::Traits
    args::ARGS
end

function MMAAtom{Traits}(args...) where {Traits <: MMATraits}
    traits = Traits()
    return MMAAtom{typeof(traits), typeof(args)}(traits, args)
end

function MMAAtom(::Type{OP}, args...) where {OP <: AbstractMMAOP}
    traits = MMATraits{OP}(args...)
    return MMAAtom{typeof(traits), typeof(args)}(traits, args)
end

function Base.show(io::IO, m::MMAAtom)
    println(io, "MMAAtom")
    println(io, "  Thread ID: ", m.traits.threadid)
    println(io, "  Layout A: ", m.traits.Alayout)
    println(io, "  Layout B: ", m.traits.Blayout)
    println(io, "  Layout C: ", m.traits.Clayout)
end
