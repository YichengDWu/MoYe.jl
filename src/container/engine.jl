abstract type Engine{T} <: DenseVector{T} end

@inline Base.IndexStyle(::Type{<:Engine}) = IndexLinear()
@inline Base.elsize(::Engine{T}) where {T} = sizeof(T)

struct ViewEngine{T, P} <: Engine{T}
    ptr::P
    len::Int
end

@inline function ViewEngine(ptr::Ref{T}, len::Int) where {T}
    return ViewEngine{T, typeof(ptr)}(ptr, len)
end

@inline function ViewEngine(A::AbstractArray)
    p = LayoutPointers.memory_reference(A)[1] # not sure what this does
    return ViewEngine(p, length(A))
end

@inline Base.pointer(A::ViewEngine) = getfield(A, :ptr)
@inline function Base.unsafe_convert(p::Type{<:Ref{T}}, A::ViewEngine{T}) where {T}
    return Base.unsafe_convert(p, pointer(A))
end

@inline Base.size(A::ViewEngine) = tuple(getfield(A, :len))
@inline Base.length(A::ViewEngine) = getfield(A, :len)

@inline function Base.getindex(A::ViewEngine{T}, i::Integer) where {T}
    @boundscheck checkbounds(A, i)
    return unsafe_load(pointer(A), i)
end

@inline function Base.setindex!(A::ViewEngine{T}, val, i::Integer) where {T}
    @boundscheck checkbounds(A, i)
    unsafe_store!(pointer(A), val, i)
    return val
end

@inline ManualMemory.preserve_buffer(::ViewEngine) = nothing

mutable struct ArrayEngine{T, L} <: DenseVector{T}
    data::NTuple{L, T}
    @inline ArrayEngine{T, L}(::UndefInitializer) where {T, L} = new{T, L}()
    @inline function ArrayEngine{T}(::UndefInitializer, ::StaticInt{L}) where {T, L}
        return ArrayEngine{T, L}(undef)
    end
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

@inline function ArrayEngine{T}(f::Function, ::StaticInt{L}) where {T, L}
    A = ArrayEngine{T, L}(undef)
    @inbounds for i in eachindex(A)
        A[i] = f(i)
    end
    return A
end

@inline function ManualMemory.preserve_buffer(A::ArrayEngine)
    return ManualMemory.preserve_buffer(getfield(A, :data))
end

Base.@propagate_inbounds function Base.getindex(A::ArrayEngine,
                                                i::Union{Integer, StaticInt})
    b = ManualMemory.preserve_buffer(A)
    GC.@preserve b begin ViewEngine(A)[i] end
end

Base.@propagate_inbounds function Base.setindex!(A::ArrayEngine, val,
                                                 i::Union{Integer, StaticInt})
    b = ManualMemory.preserve_buffer(A)
    GC.@preserve b begin ViewEngine(A)[i] = val end
end
