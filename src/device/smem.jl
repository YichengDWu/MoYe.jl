## Shared Memory, static allocation but the layout can be dynamic
@inline SharedMemory(T, ::StaticInt{L}) where {L} = CUDA.emit_shmem(T, Val(L))

function MoYeSharedArray(::Type{T}, l::StaticLayout) where {T}
    @inline
    smem = MoYe.SharedMemory(T, cosize(l))
    return MoYeArray(smem, l)
end

function MoYeSharedArray(::Type{T}, s::StaticIntTuple) where {T}
    @inline
    l = make_layout(s)
    return MoYeSharedArray(T, l)
end
