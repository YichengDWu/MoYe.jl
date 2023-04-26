struct CuTeArrayStyle{N, S, R} <: Base.Broadcast.AbstractArrayStyle{N} end

Broadcast.BroadcastStyle(::Type{<:CuTeArray{T,N,E,L}}) where {T,N,E,S,R,L<:Layout{N,S,R}} = CuTeArrayStyle{N,S,R}()
Broadcast.BroadcastStyle(::CuTeArrayStyle{N, S, R}, ::Base.Broadcast.DefaultArrayStyle{0}) where {N, S, R} = CuTeArrayStyle{N, S, R}()

# currently only defined for static layouts. Note how we instantiate the layout with types
function Base.similar(bc::Broadcast.Broadcasted{CuTeArrayStyle{N,S,R}}, ::Type{ElType}) where {ElType, N, S, R}
    return CuTeArray{ElType}(undef, Layout(make_tuple(S), make_tuple(R)))
end
