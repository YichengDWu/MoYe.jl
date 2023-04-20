
@inline function ViewEngine(ptr::LLVMPtr{T, AS}, len::IntType) where {T, AS}
    return ViewEngine{T, typeof(ptr)}(ptr, len)
end

## Shared Memory, static allocation but the layout can be dynamic
@inline SharedMemory(T, ::StaticInt{L}) = CUDA.emit_shmem(T, Val(L))
@inline function CuTeArray(ptr::LLVMPtr{T, AS.Shared}, layout::Layout) where {T, AS}
    engine = ViewEngine(ptr, cosize(layout))
    return CuTeArray(engine, layout)
end
@inline function CuTeArray(ptr::LLVMPtr{T, AS.Shared}, shape::GenIntTuple, args...) where {T<:Number, AS}
    return CuTeArray(ptr, make_layout(shape, args...))
end
