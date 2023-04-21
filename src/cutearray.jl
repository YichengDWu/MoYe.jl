"""
    CuTeArray(engine::DenseVector, layout::Layout)
    CuTeArray{T}(::UndefInitializer, layout::StaticLayout)
    CuTeArray(ptr::Ptr{T}, layout::StaticLayout)

Create a CuTeArray from an engine and a layout. See also [`ArrayEngine`](@ref) and [`ViewEngine`](@ref).

## Examples

```julia
julia> slayout = @Layout (5, 2);

julia> array_engine = ArrayEngine{Float32}(one, cosize(slayout));

julia> CuTeArray(array_engine, slayout)
5×2 CuTeArray{Float32, 2, ArrayEngine{Float32, 10}, Layout{2, Tuple{StaticInt{5}, StaticInt{2}}, Tuple{StaticInt{1}, StaticInt{5}}}} with indices static(1):static(5)×static(1):static(2):
 1.0  1.0
 1.0  1.0
 1.0  1.0
 1.0  1.0
 1.0  1.0

 julia> slayout = @Layout (5,3,2)
(static(5), static(3), static(2)):(static(1), static(5), static(15))

julia> CuTeArray{Float32}(undef, slayout) # uninitialized owning array
5×2 CuTeArray{Float32, 2, ArrayEngine{Float32, 10}, Layout{2, Tuple{Static.StaticInt{5}, Static.StaticInt{2}}, Tuple{Static.StaticInt{1}, Static.StaticInt{5}}}} with indices static(1):static(5)×static(1):static(2):
 -9.73642f-16   8.09f-43
  8.09f-43     -1.64739f13
  3.47644f36    8.09f-43
  4.5914f-41    0.0
 -9.15084f-21   0.0

julia> A = ones(10);

julia> CuTeArray(pointer(A), slayout) # create a non-owning array
5×2 CuTeArray{Float64, 2, ViewEngine{Float64, Ptr{Float64}}, Layout{2, Tuple{Static.StaticInt{5}, Static.StaticInt{2}}, Tuple{Static.StaticInt{1}, Static.StaticInt{5}}}} with indices static(1):static(5)×static(1):static(2):
 1.0  1.0
 1.0  1.0
 1.0  1.0
 1.0  1.0
 1.0  1.0

julia> function test_alloc()  # when powered by a ArrayEngine, CuTeArray is stack-allocated
    slayout = @Layout (2, 3)          # and mutable
    x = CuTeArray{Float32}(undef, slayout)
    fill!(x, 1.0f0)
    return sum(x)
end
test_alloc (generic function with 2 methods)

julia> @allocated(test_alloc())
0

```
"""
struct CuTeArray{T, N, E <: DenseVector{T}, L <: Layout{N}} <: AbstractArray{T, N}
    engine::E
    layout::L
    @inline function CuTeArray(engine::DenseVector{T}, layout::Layout{N}) where {T, N}
        return new{T, N, typeof(engine), typeof(layout)}(engine, layout)
    end
    @inline function CuTeArray(engine::DenseVector{T}, shape::GenIntTuple,
                               args...) where {T}
        return CuTeArray(engine, make_layout(shape, args...))
    end
end

@inline function CuTeArray{T}(::UndefInitializer, l::StaticLayout) where {T <: Number}
    return CuTeArray(ArrayEngine{T}(undef, cosize(l)), l)
end
@inline function CuTeArray{T}(::UndefInitializer, shape::Union{StaticInt, StaticIntTuple},
                              args...) where {T <: Number}
    l = make_layout(shape, args...)
    return CuTeArray(ArrayEngine{T}(undef, cosize(l)), l)
end
@inline function CuTeArray(ptr::Ptr{T}, layout::Layout) where {T <: Number}
    engine = ViewEngine(ptr, cosize(layout)) # this differs from the first constructor since we recompute the length
    return CuTeArray(engine, layout)
end
@inline function CuTeArray(ptr::Ptr{T}, shape::GenIntTuple, args...) where {T <: Number}
    l = make_layout(shape, args...)
    return CuTeArray(ptr, l)
end

engine(x::CuTeArray) = getfield(x, :engine)
layout(x::CuTeArray) = getfield(x, :layout)

@inline Base.size(x::CuTeArray) = Static.dynamic(map(capacity, shape(layout(x))))
@inline Base.length(x::CuTeArray) = Static.dynamic(capacity(shape(layout(x)))) # note this the logical length, not the physical length in the Engine
@inline Base.strides(x::CuTeArray) = stride(layout(x))
@inline Base.stride(x::CuTeArray, i::IntType) = getindex(stride(layout(x)), i)
@inline rank(x::CuTeArray) = rank(layout(x))
@inline depth(x::CuTeArray) = depth(layout(x))

@inline function ManualMemory.preserve_buffer(A::CuTeArray)
    return ManualMemory.preserve_buffer(engine(A))
end

# owning -> non-owning
@inline function Base.unsafe_convert(::Type{Ptr{T}},
                                     A::CuTeArray{T, N, <:ArrayEngine}) where {T, N}
    return Base.unsafe_convert(Ptr{T}, pointer_from_objref(engine(A)))
end
@inline function Base.pointer(A::CuTeArray{T, N, <:ArrayEngine}) where {N, T}
    return Base.unsafe_convert(Ptr{T}, pointer_from_objref(engine(A)))
end

@inline function Base.unsafe_convert(::Type{Ptr{T}},
                                     A::CuTeArray{T, N, <:ViewEngine}) where {T, N}
    return Base.unsafe_convert(Ptr{T}, engine(A))
end
@inline function Base.pointer(A::CuTeArray{T, N, <:ViewEngine}) where {T, N}
    return pointer(engine(A))
end

Base.@propagate_inbounds function Base.getindex(x::CuTeArray{T, N, <:ArrayEngine},
                                                ids::Union{Integer, StaticInt, IntTuple}...) where {
                                                                                                    T,
                                                                                                    N
                                                                                                    }
    b = ManualMemory.preserve_buffer(x)
    index = layout(x)(ids...)
    GC.@preserve b begin ViewEngine(engine(x))[index] end
end
Base.@propagate_inbounds function Base.getindex(x::CuTeArray,
                                                ids::Union{Integer, StaticInt, IntTuple}...)
    return getindex(engine(x), layout(x)(ids...))
end

Base.@propagate_inbounds function Base.setindex!(x::CuTeArray{T, N, <:ArrayEngine}, val,
                                                 ids::Union{Integer, StaticInt, IntTuple
                                                            }...) where {T, N}
    b = ManualMemory.preserve_buffer(x)
    index = layout(x)(ids...)
    GC.@preserve b begin ViewEngine(engine(x))[index] = val end
end
Base.@propagate_inbounds function Base.setindex!(x::CuTeArray, val,
                                                 ids::Union{Integer, StaticInt, IntTuple
                                                            }...)
    return setindex!(engine(x), val, layout(x)(ids...))
end

Base.elsize(x::CuTeArray) = Base.elsize(engine(x))
Base.sizeof(x::CuTeArray) = Base.elsize(x) * length(engine(x)) # this is the physical size

function Adapt.adapt_structure(to, x::CuTeArray)
    data = Adapt.adapt_structure(to, engine(x))
    return CuTeArray(data, layout(x))
end

function Adapt.adapt_storage(::Type{CuTeArray{T, N, A}},
                             xs::AT) where {T, N, A, AT <: AbstractArray}
    return Adapt.adapt_storage(A, xs)
end

# Array operations
# Currently don't support directly slicing, but we could make a view and then copy the view
@inline function Base.view(x::CuTeArray{T, N}, coord::Vararg{Colon, N}) where {T, N}
    b = ManualMemory.preserve_buffer(x)
    GC.@preserve b begin
        CuTeArray(pointer(x), layout(x))
    end
end
@inline function Base.view(x::CuTeArray{T},
                           coord::Union{Integer, StaticInt, IntTuple, Colon}...) where {T}
    b = ManualMemory.preserve_buffer(x)
    GC.@preserve b begin
        sliced_layout, offset = slice_and_offset(layout(x), coord)
        CuTeArray(pointer(x) + offset * sizeof(T), sliced_layout)
    end
end

@inline Base.similar(x::CuTeArray{T}) where {T} = similar(x, T)
@inline function Base.similar(x::CuTeArray, ::Type{T}) where {T}
    return CuTeArray{T}(undef, layout(x))
end
