struct Vec{T, L} <: AbstractVector{T}
    data::NTuple{L,VecElement{T}}

    Vec{T, L}(::UndefInitializer)  where {T, L} = new{T,L}()
    Vec{T, L}(x::NTuple{L, T}) where {T, L} = new{T, L}(x)
    Vec(x::NTuple{L,T}) where {T, L} = new{T, L}(x)
end

@inline @generated function Vec(x::T, y::Vararg{T, N}) where {T, N}
    L = N + 1
    quote
       Vec{T, $L}((x, y...))
    end
end

Base.IndexStyle(::Type{<:Vec}) = IndexLinear()
Base.size(::Vec{T, L}) where {T, L} = (L,)

Base.@propagate_inbounds function Base.getindex(v::Vec, i::Int)
    return v.data[i]
end

Base.@propagate_inbounds function Base.setindex(sa::Vec, val, i::Int)
    new_data = Base.setindex(sa.data, val, i)
    Vec(new_data)
end

Base.length(::Vec{T, L}) where {T, L} = L
Base.zero(::Vec{T, L}) where {T, L} = Vec(ntuple(_ -> VecElement(zero(T)), L))
