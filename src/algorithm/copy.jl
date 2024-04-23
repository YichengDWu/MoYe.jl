struct TrivialPred end
@inline Base.getindex(::TrivialPred, i) = true

function _copyto_if!(dest::NonOwningArray, src::NonOwningArray, mask)
    copy_op = select_elementwise_copy(src, dest) # would select async copy if dest is shared memory and src is global memory
    @loopinfo unroll for i in _1:size(src.layout)
        if mask[i]
            apply(copy_op, pointer(dest, i), pointer(src, i))
        end
    end
    return dest
end

function copyto_if!(dest::MoYeArray, src::MoYeArray, mask)
    @gc_preserve _copyto_if!(dest, src, mask)
end


@generated function _copyto_vec!(dest::MoYeArray{TD}, src::MoYeArray{TS}, ::Type{TV}) where {TD,TS,TV}
    if (sizeof(TD) == sizeof(TS)) && sizeof(TV) > sizeof(TD)
        return quote
            Base.@_inline_meta
            src_v = recast(TV, src)
            dest_v = recast(TV, dest)
            return _copyto_if!(dest_v, src_v, TrivialPred())
        end
    else
        return quote
            Base.@_inline_meta
            return _copyto_if!(dest, src, TrivialPred())
        end
    end
end

"""
    copyto!(dest::MoYeArray, src::MoYeArray)

Copy the contents of `src` to `dest`. The function automatically carries out potential
vectorization. In particular, while transferring data from global memory to shared memory,
it automatically initiates asynchronous copying, if your device supports so.
"""
function Base.copyto!(dest::MoYeArray, src::MoYeArray) 
    @inline 
    @gc_preserve _copyto!(dest, src)
    return dest
end

function _copyto!(dest::NonOwningArray, src::NonOwningArray)
    @inline 
    return _copyto!(dest, src, _8)
end
function _copyto!(dest::NonOwningArray{TD}, src::NonOwningArray{TS}, align::StaticInt{N}) where {TD,TS, N}
    vec_elem = max_common_vector(src, dest)
    src_bits = sizeof(TS) * 8
    #vec_bits = is_static(layout(src)) && is_static(layout(dest)) ? 
    #            min(vec_elem * src_bits, 128) : 
    #            min(vec_elem * src_bits, N)
    vec_bits = 8   # explicitly disable vectorization for now
    if vec_elem > 1 && vec_bits > 8
        return _copyto_vec!(dest, src, uint_bit(static(vec_bits)))
    else
        return _copyto_if!(dest, src, TrivialPred())
    end
end

group_tail(l::Layout{2}) = l
group_tail(l::Layout{N}) where {N} = group(l, _2, StaticInt{N}())

function generate_copy_atom_loops(dst, src, dst_layout, src_layout, n_src, n_dst, d=1)
    expr = Expr(:block)
    dst_v = Symbol(:dst_v_, d)
    src_v = Symbol(:src_v_, d)
    loop_var = Symbol(:i_, d)

    grouped_dst_layout = group_tail(dst_layout)
    grouped_src_layout = group_tail(src_layout)
    N = size(grouped_dst_layout, 2)

    push!(expr.args, :($dst_v = MoYeArray(pointer($dst), $grouped_dst_layout)))
    push!(expr.args, :($src_v = MoYeArray(pointer($src), $grouped_src_layout)))

    loop = Expr(:for, :($loop_var = _1:$N))
    loopbody = Expr(:block)
    sliced_dst = Symbol(:sliced_dst_, d)
    sliced_src = Symbol(:sliced_src_, d)
    push!(loopbody.args, :($sliced_dst = view($dst_v, :, $loop_var)))
    push!(loopbody.args, :($sliced_src = view($src_v, :, $loop_var)))

    # here we use the fact that each slice has the same layout
    sliced_layout_dst = slice(grouped_dst_layout, (:, 1))
    sliced_layout_src = slice(grouped_src_layout, (:, 1))
    if typeof(size(sliced_layout_dst)) == n_dst || typeof((sliced_layout_src)) == n_src
        push!(loopbody.args, :(copyto_unpack!(copy_atom, $sliced_dst, $sliced_src)))
    else
        new_layout_dst = sliced_layout_dst[One()]
        new_layout_src = sliced_layout_src[One()]
        push!(loopbody.args, generate_copy_atom_loops(sliced_dst, sliced_src, new_layout_dst, new_layout_src, n_src, n_dst, d+1))
    end
    push!(loopbody.args, :($(Expr(:loopinfo, (Symbol("llvm.loop.unroll.enable"), 1)))))
    push!(loop.args, loopbody)
    push!(expr.args, loop)
    return expr
end

@generated function Base.copyto!(copy_atom::AbstractCopyAtom, dst::StaticNonOwningArray, src::StaticNonOwningArray)
    expr = generate_copy_atom_loops(:dst, :src,  layout(dst)(), layout(src)(), num_val_src(copy_atom), num_val_dst(copy_atom))
    return quote
        $expr
        return dst
    end
end
function Base.copyto!(copy_atom::AbstractCopyAtom, dst::StaticNonOwningArray, src::StaticOwningArray)
    @inline
    buffer = ManualMemory.preserve_buffer(src)
    GC.@preserve buffer begin
        copyto!(copy_atom, dst, StrideArraysCore.maybe_ptr_array(src))
    end
    return dst
end
function Base.copyto!(copy_atom::AbstractCopyAtom, dst::StaticOwningArray, src::StaticNonOwningArray)
    @inline
    buffer = ManualMemory.preserve_buffer(dst)
    GC.@preserve buffer begin
        copyto!(copy_atom, StrideArraysCore.maybe_ptr_array(dst), src)
    end
    return dst
end
