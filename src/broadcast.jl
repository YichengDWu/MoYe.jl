using Base.Broadcast: AbstractArrayStyle, DefaultArrayStyle, Broadcasted
using Base.Broadcast: _broadcast_getindex, broadcast_shape
import Base.Broadcast: instantiate, materialize!, BroadcastStyle, broadcasted

struct MoYeArrayStyle{N, Shape, Stride} <: AbstractArrayStyle{N} end

BroadcastStyle(::Type{<:StaticMoYeArray{T,N,E,L}}) where {T,N,E,Shape,Stride,L<:Layout{N,Shape,Stride}} = MoYeArrayStyle{N,Shape,Stride}()
BroadcastStyle(a::MoYeArrayStyle, ::DefaultArrayStyle) = a
BroadcastStyle(a::MoYeArrayStyle{N,S}, b::MoYeArrayStyle{N,S}) where {N,S} = a 
@generated function BroadcastStyle(a::MoYeArrayStyle{N1, S1}, b::MoYeArrayStyle{N2, S2}) where {N1, S1, N2, S2}
    if N1 > N2
        return :a
    elseif N1 < N2
        return :b
    else
        if map(product, make_tuple(S1)) == map(product, make_tuple(S2))
            return :a
        else
            throw("Don't know how to broadcast MoYeArrays")
        end
    end
end

# currently only defined for static layouts
@inline function Base.similar(bc::Broadcasted{MoYeArrayStyle{N,Shape, Stride}}, 
                              ::Type{ElType}) where {ElType, N, Shape, Stride}
    return MoYeArray{ElType}(undef, make_layout_like(make_layout(make_tuple(Shape), make_tuple(Stride))))
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
