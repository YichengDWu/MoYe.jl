struct CuTeArray{T, N, E <: DenseVector{T}, L <: Layout{N}} <: AbstractArray{T, N}
    engine::E
    layout::L
    function CuTeArray(engine::DenseVector{T}, layout::Layout{N}) where {T, N}
        return new{T, N, typeof(engine), typeof(layout)}(engine, layout)
    end
end

engine(x::CuTeArray) = getfield(x, :engine)
layout(x::CuTeArray) = getfield(x, :layout)

Base.size(x::CuTeArray) = map(capacity, shape(layout(x)))
Base.length(x::CuTeArray) = length(engine(x))
Base.getindex(x::CuTeArray, ids...) = getindex(engine(x), layout(x)(ids))


function Adapt.adapt_structure(to, x::CuTeArray)
    data = Adapt.adapt_structure(to, engine(x))
    return CuTeArray(data, layout(x))
end

Adapt.adapt_storage(::Type{CuTeArray{T,N,A}}, xs::AT) where {T,N,A,AT<:AbstractArray} =
    Adapt.adapt_storage(A, xs)
