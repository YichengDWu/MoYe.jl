## Shared Memory, static allocation but the layout can be dynamic
@inline SharedMemory(T, ::StaticInt{L}) where {L} = CUDA.emit_shmem(T, Val(L))

# TODO: mimic getindex and setindex! for CuDeviceArray
