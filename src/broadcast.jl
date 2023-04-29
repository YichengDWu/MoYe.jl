struct MoyeArrayStyle{N, S, R} <: Base.Broadcast.AbstractArrayStyle{N} end

Broadcast.BroadcastStyle(::Type{<:MoyeArray{T,N,E,L}}) where {T,N,E,S,R,L<:Layout{N,S,R}} = MoyeArrayStyle{N,S,R}()
Broadcast.BroadcastStyle(::MoyeArrayStyle{N, S, R}, ::Base.Broadcast.DefaultArrayStyle{0}) where {N, S, R} = MoyeArrayStyle{N, S, R}()

# currently only defined for static layouts. Note how we instantiate the layout with types
function Base.similar(bc::Broadcast.Broadcasted{MoyeArrayStyle{N,S,R}}, ::Type{ElType}) where {ElType, N, S, R}
    return MoyeArray{ElType}(undef, Layout(make_tuple(S), make_tuple(R)))
end
