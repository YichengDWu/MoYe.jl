struct MoYeArrayStyle{N, S, R} <: Base.Broadcast.AbstractArrayStyle{N} end

Broadcast.BroadcastStyle(::Type{<:MoYeArray{T,N,E,L}}) where {T,N,E,S,R,L<:Layout{N,S,R}} = MoYeArrayStyle{N,S,R}()
Broadcast.BroadcastStyle(::MoYeArrayStyle{N, S, R}, ::Base.Broadcast.DefaultArrayStyle{0}) where {N, S, R} = MoYeArrayStyle{N, S, R}()

Broadcast.BroadcastStyle(a::MoYeArrayStyle{N,S,R}, b::MoYeArrayStyle{N,S,R}) where {N,S,R} = a
@generated function Broadcast.BroadcastStyle(a::MoYeArrayStyle{N1,S1,R1}, b::MoYeArrayStyle{N2,S2,R2}) where {N1, S1, R1, N2, S2, R2}
    if N1 > N2
        return :a
    elseif N1 < N2
        return :b
    else
        throw("Don't know how to broadcast MoYeArrays")
    end
end

# currently only defined for static layouts. Note how we instantiate the layout with types
function Base.similar(bc::Broadcast.Broadcasted{MoYeArrayStyle{N,S,R}}, ::Type{ElType}) where {ElType, N, S, R}
    return MoYeArray{ElType}(undef, Layout(make_tuple(S), make_tuple(R)))
end
