
@inline function ViewEngine(ptr::LLVMPtr{T, AS}, len::IntType) where {T, AS}
    return ViewEngine{T, typeof(ptr)}(ptr, len)
end


@inline function CuTeArray(ptr::LLVMPtr{T, A}, layout::Layout) where {T, A}
    engine = ViewEngine(ptr, cosize(layout))
    return CuTeArray(engine, layout)
end
@inline function CuTeArray(ptr::LLVMPtr{T, AS}, shape::GenIntTuple, args...) where {T, AS}
    return CuTeArray(ptr, make_layout(shape, args...))
end

## Shared Memory, static allocation but the layout can be dynamic
@inline SharedMemory(T, ::StaticInt{L}) where {L} = CUDA.emit_shmem(T, Val(L))

# TODO: mimic getindex and setindex! for CuDeviceArray
