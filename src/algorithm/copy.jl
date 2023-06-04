struct TrivialPred end
@inline Base.getindex(::TrivialPred, i) = true

function copyto_if!(dest::MoYeArray, src::MoYeArray, mask)
    copy_op = select_elementwise_copy(src, dest) # would select async copy if dest is shared memory and src is global memory
    @loopinfo unroll for i in One():size(src.layout)
        if mask[i]
            apply(copy_op, pointer(dest, i), pointer(src, i))
        end
    end
    return dest
end
function copyto_if!(copy_atom::AbstractCopyAtom, dest::MoYeArray{TD,1}, src::MoYeArray{TS,1}, mask) where {TD,TS}
    return apply(copy_atom, dest, src)
end
function copyto_if!(copy_atom::AbstractCopyAtom, dest::MoYeArray{TD,N}, src::MoYeArray{TS,N}, mask) where {TD,TS,N}
    src_v = group_modes(src, StaticInt{2}(), StaticInt{N}())
    dest_v = group_modes(dest, StaticInt{2}(), StaticInt{N}())
    @loopinfo unroll for i in One():size(src_v.layout, 2)
        if mask[i]
            apply(copy_atom, view(dest_v, :, i), view(src_v, :, i))
        end
    end
    return dest
end

@inline function copyto_vec!(dest::MoYeArray{TD}, src::MoYeArray{TS}, ::Type{TV}) where {TD,TS,TV}
    if (sizeof(TD) == sizeof(TS)) && sizeof(TV) > sizeof(TD)
        src_v = recast(TV, src)
        dest_v = recast(TV, dest)
        #print("Vectorized copyto! from $(sizeof(TS)) bytes to $(sizeof(TV)) bytes")
        copyto_if!(dest_v, src_v, TrivialPred())
    else
        copyto_if!(dest, src, TrivialPred())
    end
    return dest
end

"""
    copyto!(dest::StaticNonOwningArray, src::StaticNonOwningArray)

Copy the contents of `src` to `dest`. The function automatically carries out potential
vectorization. In particular, while transferring data from global memory to shared memory,
it automatically initiates asynchronous copying, if your device supports so.

!!! note
    It should be used with @gc_preserve if `dest` or `src` is powered by an ArrayEngine.
"""
function Base.copyto!(dest::StaticNonOwningArray{TD}, src::StaticNonOwningArray{TS}) where {TD,TS}
    N = max_common_vector(src, dest)
    if N ≤ 1
        return copyto_if!(dest, src, TrivialPred())
    else
        vec_bits = N * sizeof(TS) * 8
        TV = uint_bit(static(min(128, vec_bits)))
        return copyto_vec!(dest, src, TV)
    end
    return dest
end

function Base.copyto!(copy_atom::AbstractCopyAtom, dest::MoYeArray, src::MoYeArray)
    return copyto_if!(copy_atom, dest, src, TrivialPred())
end
