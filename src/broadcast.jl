using Base.Broadcast: AbstractArrayStyle, DefaultArrayStyle, Broadcasted
import Base.Broadcast: instantiate, materialize!, BroadcastStyle

struct MoYeArrayStyle{N, S, R} <: AbstractArrayStyle{N} end

BroadcastStyle(::Type{<:MoYeArray{T,N,E,L}}) where {T,N,E,S,R,L<:Layout{N,S,R}} = MoYeArrayStyle{N,S,R}()
BroadcastStyle(::MoYeArrayStyle{N, S, R}, ::DefaultArrayStyle{0}) where {N, S, R} = MoYeArrayStyle{N, S, R}()

BroadcastStyle(a::MoYeArrayStyle{N,S,R}, b::MoYeArrayStyle{N,S,R}) where {N,S,R} = a
@generated function BroadcastStyle(a::MoYeArrayStyle{N1,S1,R1}, b::MoYeArrayStyle{N2,S2,R2}) where {N1, S1, R1, N2, S2, R2}
    if N1 > N2
        return :a
    elseif N1 < N2
        return :b
    else
        throw("Don't know how to broadcast MoYeArrays")
    end
end

# currently only defined for static layouts. Note how we instantiate the layout with types
@inline function Base.similar(bc::Broadcasted{MoYeArrayStyle{N,S,R}}, ::Type{ElType}) where {ElType, N, S, R}
    return MoYeArray{ElType}(undef, Layout(make_tuple(S), make_tuple(R)))
end

@inline function Base.copyto!(dest::MoYeArray{T,N,A,L}, bc::Broadcasted{MoYeArrayStyle{N,S,R}}) where {T,N,S,R,A<:ArrayEngine,L<:Layout{N,S,R}}
    @gc_preserve copyto!(dest, bc)
    return dest
end
