"""
    CuTeArray(engine::DenseVector, layout::Layout)

Create a CuTeArray from an engine and a layout. See also [`ArrayEngine`](@ref) and [`ViewEngine`](@ref).

## Examples

```julia
julia> slayout = @Layout (5, 2);

julia> array_engine = ArrayEngine{Float32}(one, static(10));

julia> CuTeArray(array_engine, slayout)
5×2 CuTeArray{Float32, 2, ArrayEngine{Float32, 10}, Layout{2, Tuple{StaticInt{5}, StaticInt{2}}, Tuple{StaticInt{1}, StaticInt{5}}}} with indices static(1):static(5)×static(1):static(2):
 1.0  1.0
 1.0  1.0
 1.0  1.0
 1.0  1.0
 1.0  1.0
```
"""
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

@inline function ManualMemory.preserve_buffer(A::CuTeArray)
    return ManualMemory.preserve_buffer(engine(A))
end

@inline function Base.unsafe_convert(::Type{Ptr{T}},
                                     A::CuTeArray{T, N, <:ArrayEngine}) where {T, N}
    return Base.unsafe_convert(Ptr{T}, pointer_from_objref(engine(A)))
end
@inline function Base.pointer(A::CuTeArray{T, N, <:ArrayEngine}) where {N, T}
    return Base.unsafe_convert(Ptr{T}, pointer_from_objref(engine(A)))
end

Base.@propagate_inbounds function Base.getindex(x::CuTeArray{T, N, <:ArrayEngine},
                                                ids::Union{Integer, StaticInt}...) where {T,
                                                                                          N}
    b = ManualMemory.preserve_buffer(x)
    index = layout(x)(ids...)
    GC.@preserve b begin ViewEngine(engine(x))[index] end
end
Base.@propagate_inbounds function Base.getindex(x::CuTeArray,
                                                ids::Union{Integer, StaticInt}...)
    return getindex(engine(x), layout(x)(ids...))
end

# TODO: support slicing

Base.@propagate_inbounds function Base.setindex!(x::CuTeArray{T, N, <:ArrayEngine}, val,
                                                 ids::Union{Integer, StaticInt}...) where {
                                                                                           T,
                                                                                           N
                                                                                           }
    b = ManualMemory.preserve_buffer(x)
    index = layout(x)(ids...)
    GC.@preserve b begin ViewEngine(engine(x))[index] = val end
end
Base.@propagate_inbounds function Base.setindex!(x::CuTeArray, val,
                                                 ids::Union{Integer, StaticInt}...)
    return setindex!(engine(x), val, layout(x)(ids...))
end

Base.elsize(x::CuTeArray) = Base.elsize(engine(x))
Base.sizeof(x::CuTeArray) = Base.elsize(x) * length(x)

function Adapt.adapt_structure(to, x::CuTeArray)
    data = Adapt.adapt_structure(to, engine(x))
    return CuTeArray(data, layout(x))
end

function Adapt.adapt_storage(::Type{CuTeArray{T, N, A}},
                             xs::AT) where {T, N, A, AT <: AbstractArray}
    return Adapt.adapt_storage(A, xs)
end
