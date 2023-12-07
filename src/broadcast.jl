using Base.Broadcast: AbstractArrayStyle, DefaultArrayStyle, Broadcasted
using Base.Broadcast: _broadcast_getindex, broadcast_shape
import Base.Broadcast: instantiate, materialize!, BroadcastStyle, broadcasted

struct MoYeArrayStyle{N, S} <: AbstractArrayStyle{N} end

BroadcastStyle(::Type{<:StaticMoYeArray{T,N,E,L}}) where {T,N,E,S,L<:Layout{N,S}} = MoYeArrayStyle{N,S}()
BroadcastStyle(a::MoYeArrayStyle, ::DefaultArrayStyle) = a

BroadcastStyle(a::MoYeArrayStyle{N,S}, b::MoYeArrayStyle{N,S}) where {N,S} = a
@generated function BroadcastStyle(a::MoYeArrayStyle{N1,S1}, b::MoYeArrayStyle{N2,S2}) where {N1, S1, N2, S2}
    if N1 > N2
        return :a
    elseif N1 < N2
        return :b
    else
        throw("Don't know how to broadcast MoYeArrays")
    end
end

# currently only defined for static size
@inline function Base.similar(bc::Broadcasted{MoYeArrayStyle{N,S}}, ::Type{ElType}) where {ElType, N, S}
    return MoYeArray{ElType}(undef, make_tuple(S))
end

@inline function Base.copyto!(dest::StaticOwningArray{T,N,L}, bc::Broadcasted{MoYeArrayStyle{N,S}}) where {T,N,S,L<:Layout{N,S}}
    @gc_preserve copyto!(dest, bc)
    return dest
end

# have to define these to avoid ambiguities...
@inline broadcasted(f::Function, x::StaticOwningArray) = @gc_preserve broadcasted(f, x)
@inline broadcasted(f::Function, x::StaticOwningArray, y::StaticOwningArray) = @gc_preserve broadcasted(f, x, y)
@inline broadcasted(f::Function, x::StaticOwningArray, y::Number) = @gc_preserve broadcasted(f, x, y)
@inline broadcasted(f::Function, x::Number, y::StaticOwningArray) = @gc_preserve broadcasted(f, x, y)
@inline broadcasted(f::Function, x::StaticOwningArray, y::StaticNonOwningArray) = @gc_preserve broadcasted(f, x, y)
@inline broadcasted(f::Function, x::StaticNonOwningArray, y::StaticOwningArray) = @gc_preserve broadcasted(f, x, y)
