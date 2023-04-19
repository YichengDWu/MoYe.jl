struct CuTeArray{T, N, A<:AbstractVector{T}, Layout{N}} <: AbstractArray{T, N}
    data::A
    layout::Layout
    function CuTeArray(data::AbstractVector{T}, layout::Layout{N}) where {T, N}
        return new{T, N, typeof(data), typeof(layout)}(data, layout)
    end
end

get_data(x::CuTeArray) = x.data
get_layout(x::CuTeArray) = x.layout

function Adapt.adapt_structure(to, x::CuTeArray)
    data = Adapt.adapt_structure(to, get_data(x))
    return CuTe(data, get_layout(x))
end

Adapt.adapt_storage(::Type{CuTeArray{T,N,A}}, xs::AT) where {T,N,A,AT<:AbstractArray} =
    Adapt.adapt_storage(A, xs)
