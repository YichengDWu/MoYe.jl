struct CuTeArray{T, N, E <: Engine{T}, L <: Layout{N}} <: AbstractArray{T, N}
    engine::E
    layout::L
    function CuTeArray(engine::Engine{T}, layout::Layout{N}) where {T, N}
        return new{T, N, typeof(engine), typeof(layout)}(engine, layout)
    end
end

engine(x::CuTeArray) = getfield(x, :engine)
layout(x::CuTeArray) = getfield(x, :layout)

#function Adapt.adapt_structure(to, x::CuTeArray)
#    data = Adapt.adapt_structure(to, engine(x))
#    return CuTeArray(data, layout(x))
#end

#Adapt.adapt_storage(::Type{CuTeArray{T,N,A}}, xs::AT) where {T,N,A,AT<:AbstractArray} =
#    Adapt.adapt_storage(A, xs)
