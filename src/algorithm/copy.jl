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

@generated function copyto_vec!(dest::MoYeArray{TD}, src::MoYeArray{TS}, ::Type{TV}) where {TD,TS,TV}
    if (sizeof(TD) == sizeof(TS)) && sizeof(TV) > sizeof(TD)
        return quote
            src_v = recast(TV, src)
            dest_v = recast(TV, dest)
            #print("Vectorized copyto! from $(sizeof(TS)) bytes to $(sizeof(TV)) bytes")
            copy_op = select_elementwise_copy(src_v, dest_v)
            @loopinfo unroll for i in One():size(src_v.layout)
                apply(copy_op, pointer(dest_v, i), pointer(src_v, i))
            end
            return dest
        end
    else
        return quote
            copy_op = select_elementwise_copy(src, dest)
            @loopinfo unroll for i in One():size(src.layout)
                apply(copy_op, pointer(dest, i), pointer(src, i))
            end
            return dest
        end
    end
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
    if N == One() || N == Zero()
        return copyto_if!(dest, src, TrivialPred())
    else
        vec_bits = N * sizeof(TS) * 8
        TV = uint_bit(static(min(128, vec_bits)))
        return copyto_vec!(dest, src, TV)
    end
    return dest
end

Base.@assume_effects :terminates_globally function Base.copyto!(copy_atom::AbstractCopyAtom, dest::MoYeArray{TD,1}, src::MoYeArray{TS,1}) where {TD,TS}
    apply(copy_atom, dest, src)
    return dest
end
Base.@assume_effects :terminates_globally function Base.copyto!(copy_atom::AbstractCopyAtom, dest::MoYeArray{TD,N}, src::MoYeArray{TS,N}) where {TD,TS,N}
    src_v = group_modes(src, StaticInt{2}(), StaticInt{N}())
    dest_v = group_modes(dest, StaticInt{2}(), StaticInt{N}())
    @loopinfo unroll for i in One():size(src_v.layout, 2)
        apply(copy_atom, view(dest_v, :, i), view(src_v, :, i))
    end
    return dest
end
#=
Base.@assume_effects :terminates_globally function Base.copyto!(copy_atom::AbstractCopyAtom, dst::MoYeArray{TD,N}, src::MoYeArray{TS,N}) where {TD,TS,N}
    expr = Expr(:block)
    n_src = num_val_src(copy_atom)
    n_dst = num_val_dst(copy_atom)
    src_layout = layout(src)
    dst_layout = layout(dst)

    function inner(dst, src, src_layout, dst_layout)
        dst_v = gensym(:dst_v)
        src_v = gensym(:src_v)
        loop_var = gensym(:i)

        grouped_dst_layout = group_tail(dst_layout)()
        grouped_src_layout = group_tail(src_layout)()
        N = size(grouped_dst_layout, 2)

        push!(expr.args, :($dst_v = MoYeArray(pointer($dst), $grouped_dst_layout)))
        push!(expr.args, :($src_v = MoYeArray(pointer($src), $grouped_src_layout)))

        loopbody = Expr(:for, :($loop_var in 1:$N))
        slice_dst = gensym(:slice_dst)
        slice_src = gensym(:slice_src)
        push!(loopbody.args, :($slice_dst = view($dst_v, :, $loop_var)))
        push!(loopbody.args, :($slice_src = view($src_v, :, $loop_var)))

        if done
            push!(loopbody.args, :(copyto_unpack!(copy_atom, $slice_dst, $slice_src)))
        else
            push!(loopbody.args, inner(slice_dst, slice_src))
        end
        return Expr(:macrocall, Symbol("@loopinfo"), :unroll, loopbody)
    end

    return dst
end


expr = Expr(:tuple)
function repeat_inner(expr, T)
    for i in T.parameters
        if i <: IntType
            push!(expr.args, :x)
        elseif i <: Tuple
            push!(expr.args, repeat_inner(Expr(:tuple), i))
        end
    end
    return expr
end
repeat_inner(expr, T)
Base.@assume_effects :terminates_globally @generated function apply(copy_atom::CP, dst::StaticMoYeArray{TD, 1},
                          src::StaticMoYeArray{TS, 1}) where {CP <: AbstractCopyAtom, TD, TS
                                                              }
    if num_val_src(copy_atom) == size(layout(src)) ||
       num_val_dst(copy_atom) == size(layout(dst))
        return quote
            Base.@_inline_meta
            copyto_unpack!(copy_atom, dst, src)
        end
    elseif shape(layout(src)) <: Tuple && shape(layout(dst)) <: Tuple
        dst_layout = layout(dst)[One()]()
        src_layout = layout(src)[One()]()
        return quote
            Base.@_inline_meta
            copyto!(copy_atom, MoYeArray(pointer(dst), $dst_layout),
                    MoYeArray(pointer(src), $src_layout))
        end
    else
        throw(ArgumentError("Cannot copy from $src to $dst, $(num_val_src(copy_atom))!= $(size(layout(src)))
                             $(shape(layout(src)) <: Tuple) is not a tuple, and $(shape(layout(dst)) <: Tuple) "))
    end
end
=#
