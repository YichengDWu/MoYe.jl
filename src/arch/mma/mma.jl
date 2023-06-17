abstract type PTXOperation end

@inline apply(op::PTXOperation, args...) = op(args...)

abstract type AbstractMMAOP{DRegisters, ARegisters, BRegisters, CRegisters} <: PTXOperation end

@inline Adapt.adapt(to, x::AbstractMMAOP) = x

@inline fma!(mmaop::AbstractMMAOP, a, b, c) = fma!(mmaop, c, a, b, c)

"""
  Registers{T,S}

A struct that wraps the register file type `T` and number of register files `S`.
"""
struct Registers{T,S} end

@inline Base.eltype(::Registers{T}) where {T} = T
@inline Base.length(::Registers{T, L}) where {T, L} = L
@inline Base.eltype(::Type{<:Registers{T}}) where {T} = T
@inline Base.length(::Type{Registers{T, L}}) where {T, L} = L

function Base.getproperty(obj::AbstractMMAOP{DRegisters, ARegisters, BRegisters, CRegisters},
                          sym::Symbol) where {DRegisters, ARegisters,
                                              BRegisters, CRegisters}
    if sym === :DRegisters
        return DRegisters
    elseif sym === :ARegisters
        return ARegisters
    elseif sym === :BRegisters
        return BRegisters
    elseif sym === :CRegisters
        return CRegisters
    else
        return getfield(obj,sym)
    end
end

function Base.propertynames(::AbstractMMAOP)
    return (:DRegisters, :ARegisters, :BRegisters, :CRegisters)
end

@inline regtype_d(mma_op::AbstractMMAOP) = eltype(mma_op.DRegisters)
@inline regtype_a(mma_op::AbstractMMAOP) = eltype(mma_op.ARegisters)
@inline regtype_b(mma_op::AbstractMMAOP) = eltype(mma_op.BRegisters)
@inline regtype_c(mma_op::AbstractMMAOP) = eltype(mma_op.CRegisters)

@inline regnum_d(mma_op::AbstractMMAOP) = length(mma_op.DRegisters)
@inline regnum_a(mma_op::AbstractMMAOP) = length(mma_op.ARegisters)
@inline regnum_b(mma_op::AbstractMMAOP) = length(mma_op.BRegisters)
@inline regnum_c(mma_op::AbstractMMAOP) = length(mma_op.CRegisters)

struct UniversalFMA{D,A,B,C} <: AbstractMMAOP{Registers{D, 1}, Registers{A, 1},
    Registers{B, 1}, Registers{C, 1}}
end

@inline fma!(::UniversalFMA, d, a, b, c) = d .= a .* b .+ c
