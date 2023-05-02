"""
    ViewEngine{T, P}

A wrapper of a pointer. `P` is the type of the pointer.
"""
struct ViewEngine{T, P}
    ptr::P
end

@inline function ViewEngine(ptr::Ptr{T}) where {T}
    return ViewEngine{T, typeof(ptr)}(ptr)
end
@inline function ViewEngine(ptr::LLVMPtr{T, AS}) where {T, AS}
    return ViewEngine{T, typeof(ptr)}(ptr)
end

@inline function ViewEngine(A::AbstractArray)
    p = LayoutPointers.memory_reference(A)[1]
    return ViewEngine(p)
end

@inline function ViewEngine(A::ViewEngine)
    return A
end

@inline Base.pointer(A::ViewEngine) = getfield(A, :ptr)
@inline function Base.unsafe_convert(p::Type{Ptr{T}}, A::ViewEngine{T}) where {T}
    return Base.unsafe_convert(p, pointer(A))
end
@inline function Base.unsafe_convert(p::Type{LLVMPtr{T, AS}}, A::ViewEngine{T}) where {T, AS}
    return Base.unsafe_convert(p, pointer(A))
end

@inline function Base.getindex(A::ViewEngine{T, <:LLVMPtr{T}}, i::Integer) where {T}
    align = Base.datatype_alignment(T)
    return unsafe_load(pointer(A), i, Val(align))
end
@inline function Base.getindex(A::ViewEngine{T}, i::Integer) where {T}
    return unsafe_load(pointer(A), i)
end

@inline function Base.setindex!(A::ViewEngine{T, <:LLVMPtr{T}}, val, i::Integer) where {T}
    align = Base.datatype_alignment(T)
    return unsafe_store!(pointer(A), val, i, Val(align))
end
@inline function Base.setindex!(A::ViewEngine{T}, val, i::Integer) where {T}
    return unsafe_store!(pointer(A), val, i)
end

@inline ManualMemory.preserve_buffer(::ViewEngine) = nothing

"""
    ArrayEngine{T, L} <: DenseVector{T}

A owning and mutable vector of type `T` with static length `L`.

## Examples

```julia
julia> x = ArrayEngine{Float32}(undef, static(3))
3-element ArrayEngine{Float32, 3}:
 -9.8271385f-36
  7.57f-43
 -9.8271385f-36

julia> x[1] = 10f0
10.0f0

julia> x
3-element ArrayEngine{Float32, 3}:
 10.0
  7.57f-43
 -9.8271385f-36
"""
mutable struct ArrayEngine{T, L} <: DenseVector{T}
    data::NTuple{L, T}
    @inline ArrayEngine{T, L}(::UndefInitializer) where {T, L} = new{T, L}()
    @inline function ArrayEngine{T}(::UndefInitializer, ::StaticInt{L}) where {T, L}
        return ArrayEngine{T, L}(undef)
    end
    @inline ArrayEngine(data::NTuple{L, T}) where {T, L} = new{T, L}(data)
end

@inline function Base.unsafe_convert(::Type{Ptr{T}}, A::ArrayEngine{T}) where {T}
    return Base.unsafe_convert(Ptr{T}, pointer_from_objref(A))
end
@inline function Base.pointer(A::ArrayEngine{T}) where {T}
    return Base.unsafe_convert(Ptr{T}, pointer_from_objref(A))
end

@inline Base.size(::ArrayEngine{T, L}) where {T, L} = (L,)
@inline Base.length(::ArrayEngine{T, L}) where {T, L} = L
@inline Base.similar(::ArrayEngine{T, L}) where {T, L} = ArrayEngine{T, L}(undef)
@inline function Base.similar(A::ArrayEngine, ::Type{T}) where {T}
    return ArrayEngine{T}(undef, static(length(A)))
end

@inline function ArrayEngine{T}(f::Function, ::StaticInt{L}) where {T, L} # not very useful
    A = ArrayEngine{T, L}(undef)
    @loopinfo unroll for i in eachindex(A)
        @inbounds A[i] = f(i)
    end
    return A
end

@inline function ManualMemory.preserve_buffer(A::ArrayEngine)
    return ManualMemory.preserve_buffer(getfield(A, :data))
end

@inline function Base.getindex(A::ArrayEngine, i::Integer)
    @boundscheck checkbounds(A, i)
    b = ManualMemory.preserve_buffer(A)
    GC.@preserve b begin ViewEngine(A)[i] end
end

@inline function Base.setindex!(A::ArrayEngine, val, i::Integer)
    @boundscheck checkbounds(A, i)
    b = ManualMemory.preserve_buffer(A)
    GC.@preserve b begin ViewEngine(A)[i] = val end
end


const Engine{T} = Union{ViewEngine{T}, ArrayEngine{T}}
