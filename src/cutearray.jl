"""
    MoyeArray(engine::DenseVector, layout::Layout)
    MoyeArray{T}(::UndefInitializer, layout::StaticLayout)
    MoyeArray(ptr::Ptr{T}, layout::StaticLayout)

Create a MoyeArray from an engine and a layout. See also [`ArrayEngine`](@ref) and [`ViewEngine`](@ref).

## Examples

```julia
julia> slayout = @Layout (5, 2);

julia> array_engine = ArrayEngine{Float32}(one, cosize(slayout));

julia> MoyeArray(array_engine, slayout)
5×2 MoyeArray{Float32, 2, ArrayEngine{Float32, 10}, Layout{2, Tuple{StaticInt{5}, StaticInt{2}}, Tuple{StaticInt{1}, StaticInt{5}}}} with indices static(1):static(5)×static(1):static(2):
 1.0  1.0
 1.0  1.0
 1.0  1.0
 1.0  1.0
 1.0  1.0

 julia> slayout = @Layout (5,3,2)
(static(5), static(3), static(2)):(static(1), static(5), static(15))

julia> MoyeArray{Float32}(undef, slayout) # uninitialized owning array
5×2 MoyeArray{Float32, 2, ArrayEngine{Float32, 10}, Layout{2, Tuple{Static.StaticInt{5}, Static.StaticInt{2}}, Tuple{Static.StaticInt{1}, Static.StaticInt{5}}}} with indices static(1):static(5)×static(1):static(2):
 -9.73642f-16   8.09f-43
  8.09f-43     -1.64739f13
  3.47644f36    8.09f-43
  4.5914f-41    0.0
 -9.15084f-21   0.0

julia> A = ones(10);

julia> MoyeArray(pointer(A), slayout) # create a non-owning array
5×2 MoyeArray{Float64, 2, ViewEngine{Float64, Ptr{Float64}}, Layout{2, Tuple{Static.StaticInt{5}, Static.StaticInt{2}}, Tuple{Static.StaticInt{1}, Static.StaticInt{5}}}} with indices static(1):static(5)×static(1):static(2):
 1.0  1.0
 1.0  1.0
 1.0  1.0
 1.0  1.0
 1.0  1.0

julia> function test_alloc()  # when powered by a ArrayEngine, MoyeArray is stack-allocated
    slayout = @Layout (2, 3)          # and mutable
    x = MoyeArray{Float32}(undef, slayout)
    fill!(x, 1.0f0)
    return sum(x)
end
test_alloc (generic function with 2 methods)

julia> @allocated(test_alloc())
0

```
"""
struct MoyeArray{T, N, E <: DenseVector{T}, L <: Layout{N}} <: AbstractArray{T, N}
    engine::E
    layout::L
    @inline function MoyeArray(engine::DenseVector{T}, layout::Layout{N}) where {T, N}
        return new{T, N, typeof(engine), typeof(layout)}(engine, layout)
    end
    @inline function MoyeArray(engine::DenseVector{T}, shape::GenIntTuple,
                               args...) where {T}
        return MoyeArray(engine, make_layout(shape, args...))
    end
end

@inline function MoyeArray{T}(::UndefInitializer, l::StaticLayout) where {T}
    return MoyeArray(ArrayEngine{T}(undef, cosize(l)), l)
end
@inline function MoyeArray{T}(::UndefInitializer, shape::Union{StaticInt, StaticIntTuple},
                              args...) where {T}
    l = make_layout(shape, args...)
    return MoyeArray(ArrayEngine{T}(undef, cosize(l)), l)
end
@inline function MoyeArray(ptr::Ptr{T}, layout::Layout) where {T}
    engine = ViewEngine(ptr, cosize(layout)) # this differs from the first constructor since we recompute the length
    return MoyeArray(engine, layout)
end
@inline function MoyeArray(ptr::Ptr{T}, shape::GenIntTuple, args...) where {T <: Number}
    l = make_layout(shape, args...)
    return MoyeArray(ptr, l)
end
@inline function MoyeArray(ptr::LLVMPtr{T, A}, layout::Layout) where {T, A}
    engine = ViewEngine(ptr, cosize(layout))
    return MoyeArray(engine, layout)
end
@inline function MoyeArray(ptr::LLVMPtr{T, AS}, shape::GenIntTuple, args...) where {T, AS}
    return MoyeArray(ptr, make_layout(shape, args...))
end

const BitMoyeArray{N, E, L} = MoyeArray{Bool, N, E, L}

engine(x::MoyeArray) = getfield(x, :engine)
layout(x::MoyeArray) = getfield(x, :layout)

@inline Base.size(x::MoyeArray) = tuple(Static.dynamic(map(capacity, shape(layout(x))))...)
@inline Base.length(x::MoyeArray) = Static.dynamic(capacity(shape(layout(x)))) # note this the logical length, not the physical length in the Engine
@inline Base.strides(x::MoyeArray) = stride(layout(x))
@inline Base.stride(x::MoyeArray, i::IntType) = getindex(stride(layout(x)), i)
@inline rank(x::MoyeArray) = rank(layout(x))
@inline depth(x::MoyeArray) = depth(layout(x))

@inline function ManualMemory.preserve_buffer(A::MoyeArray)
    return ManualMemory.preserve_buffer(engine(A))
end

@inline function Base.unsafe_convert(::Type{Ptr{T}},
                                     A::MoyeArray{T}) where {T}
    return Base.unsafe_convert(Ptr{T}, engine(A))
end

@inline function Base.pointer(A::MoyeArray)
    return pointer(engine(A))
end

"""
    pointer(A::MoyeArray, i::Integer)

Return a pointer to the element at the logical index `i` in `A`, not the physical index.
"""
@inline function Base.pointer(x::MoyeArray{T}, i::Integer) where {T}
    idx = x.layout(convert(Int, i))
    return pointer(x) + (idx-one(idx))*sizeof(T)
end

Base.IndexStyle(::Type{<:MoyeArray}) = IndexLinear()

Base.@propagate_inbounds function Base.getindex(x::MoyeArray{T, N, <:ArrayEngine},
                                                ids::Union{Integer, StaticInt, IntTuple}...) where {
                                                                                                    T,
                                                                                                    N
                                                                                                    }
    b = ManualMemory.preserve_buffer(x)
    index = layout(x)(ids...)
    GC.@preserve b begin ViewEngine(engine(x))[index] end
end
Base.@propagate_inbounds function Base.getindex(x::MoyeArray,
                                                ids::Union{Integer, StaticInt, IntTuple}...)
    return getindex(engine(x), layout(x)(ids...))
end

Base.@propagate_inbounds function Base.setindex!(x::MoyeArray{T, N, <:ArrayEngine}, val,
                                                 ids::Union{Integer, StaticInt, IntTuple
                                                            }...) where {T, N}
    b = ManualMemory.preserve_buffer(x)
    index = layout(x)(ids...)
    GC.@preserve b begin ViewEngine(engine(x))[index] = val end
end
Base.@propagate_inbounds function Base.setindex!(x::MoyeArray, val,
                                                 ids::Union{Integer, StaticInt, IntTuple
                                                            }...)
    return setindex!(engine(x), val, layout(x)(ids...))
end

Base.elsize(x::MoyeArray) = Base.elsize(engine(x))
Base.sizeof(x::MoyeArray) = Base.elsize(x) * length(engine(x)) # this is the physical size

function Adapt.adapt_structure(to, x::MoyeArray)
    data = Adapt.adapt_structure(to, engine(x))
    return MoyeArray(data, layout(x))
end

function Adapt.adapt_storage(::Type{MoyeArray{T, N, A}},
                             xs::AT) where {T, N, A, AT <: AbstractArray}
    return Adapt.adapt_storage(A, xs)
end

@inline StrideArraysCore.maybe_ptr_array(A::MoyeArray) = MoyeArray(pointer(A), layout(A))

# Array operations
# Currently don't support directly slicing, but we could make a view and then copy the view
@inline function Base.view(x::MoyeArray{T, N}, coord::Vararg{Colon, N}) where {T, N}
    b = ManualMemory.preserve_buffer(x)
    GC.@preserve b begin
        MoyeArray(pointer(x), layout(x))
    end
end
@inline function Base.view(x::MoyeArray{T},
                           coord...) where {T}
    b = ManualMemory.preserve_buffer(x)
    GC.@preserve b begin
        sliced_layout, offset = slice_and_offset(layout(x), coord)
        MoyeArray(pointer(x) + offset * sizeof(T), sliced_layout)
    end
end

@inline function Base.similar(x::MoyeArray{S,N,E,<:StaticLayout}, ::Type{T}) where {S,N,E,T}
    return MoyeArray{T}(undef, layout(x))
end


"""
    recast(::Type{NewType}, x::MoyeArray{OldType}) -> MoyeArray{NewType}

Recast the element type of a MoyeArray. This is similar to `Base.reinterpret`, but dose all
the computation at compile time, if possible.

## Examples
```julia
julia> x = MoyeArray{Int32}(undef, @Layout((2,3)))
2×3 MoyeArray{Int32, 2, ArrayEngine{Int32, 6}, Layout{2, Tuple{Static.StaticInt{2}, Static.StaticInt{3}}, Tuple{Static.StaticInt{1}, Static.StaticInt{2}}}}:
 -1948408944           0  2
         514  -268435456  0

julia> x2 = recast(Int16, x)
4×3 MoyeArray{Int16, 2, ViewEngine{Int16, Ptr{Int16}}, Layout{2, Tuple{Static.StaticInt{4}, Static.StaticInt{3}}, Tuple{Static.StaticInt{1}, Static.StaticInt{4}}}}:
 -23664      0  2
 -29731      0  0
    514      0  0
      0  -4096  0

julia> x3 = recast(Int64, x)
1×3 MoyeArray{Int64, 2, ViewEngine{Int64, Ptr{Int64}}, Layout{2, Tuple{Static.StaticInt{1}, Static.StaticInt{3}}, Tuple{Static.StaticInt{1}, Static.StaticInt{1}}}}:
 2209959748496  -1152921504606846976  2
```
"""
@inline function recast(::Type{NewType}, x::MoyeArray{OldType}) where {NewType, OldType}
    b = ManualMemory.preserve_buffer(x)
    GC.@preserve b begin
        old_layout = layout(x)
        new_layout = recast(old_layout, NewType, OldType)
        if sizeof(OldType) < sizeof(NewType) # TODO: handle composed layout
            shape_diff = map(-, flatten(shape(old_layout)), flatten(shape(new_layout)))
            extent_diff = map(*, shape_diff, flatten(stride(old_layout)))
            offset = foldl((i,a)->i+min(a, Zero()), extent_diff; init=Zero())
            return MoyeArray(recast(NewType, pointer(x) + offset * sizeof(OldType)), new_layout)
        else
            return MoyeArray(recast(NewType, pointer(x)), new_layout)
        end
    end
end
