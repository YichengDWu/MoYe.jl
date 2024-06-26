struct TrivialPred end
@inline (::TrivialPred)(i) = true

function _copyto_if!(dest::NonOwningArray, src::NonOwningArray, mask)
    copy_op = select_elementwise_copy(src, dest) # would select async copy if dest is shared memory and src is global memory
    @loopinfo unroll for i in _1:size(src.layout)
        if mask(i)
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

@inline group_tail(l::Layout{2}) = l
@inline group_tail(l::Layout{N}) where {N} = group(l, _2, StaticInt{N}())
@inline group_tail(l::Tuple{Vararg{Union{IntType, Tuple}, 2}}) = l
@inline group_tail(l::Tuple{Vararg{Union{IntType, Tuple}, N}}) where {N} = (Base.first(l), Base.tail(l))

function generate_copy_atom_loops(dst, src, dst_shape, src_shape, n_src, n_dst, d=1)
    expr = Expr(:block)
    dst_v = Symbol(:dst_v_, d)
    src_v = Symbol(:src_v_, d)
    loop_var = Symbol(:i_, d)

    grouped_dst_shape = group_tail(dst_shape)
    grouped_src_shape = group_tail(src_shape)

    push!(expr.args, :($dst_v = MoYeArray(pointer($dst), group_tail(layout($dst)[_1]))))
    push!(expr.args, :($src_v = MoYeArray(pointer($src), group_tail(layout($src)[_1]))))

    loop = Expr(:for, :($loop_var = _1:$(product(grouped_dst_shape[2]))))
    loopbody = Expr(:block)
    sliced_dst = Symbol(:sliced_dst_, d)
    sliced_src = Symbol(:sliced_src_, d)
    push!(loopbody.args, :($sliced_dst = view($dst_v, :, $loop_var)))
    push!(loopbody.args, :($sliced_src = view($src_v, :, $loop_var)))

    # here we use the fact that each slice has the same layout
    sliced_layout_dst = slice(grouped_dst_shape, (:, 1))
    sliced_layout_src = slice(grouped_src_shape, (:, 1))
    if typeof(product(grouped_dst_shape[1])) == n_dst || typeof(product(grouped_src_shape[1])) == n_src
        push!(loopbody.args, :(copyto_unpack!(copy_atom, $sliced_dst, $sliced_src)))
    else
        new_layout_dst = grouped_dst_shape[1]
        new_layout_src = grouped_src_shape[1]
        push!(loopbody.args, generate_copy_atom_loops(sliced_dst, sliced_src, new_layout_dst, new_layout_src, n_src, n_dst, d+1))
    end
    push!(loopbody.args, :($(Expr(:loopinfo, (Symbol("llvm.loop.unroll.enable"), 1)))))
    push!(loop.args, loopbody)
    push!(expr.args, loop)
    return expr
end

function Base.copyto!(copy_atom::AbstractCopyAtom, dst::MoYeArray, src::MoYeArray)
    @inline
    @gc_preserve _copyto!(copy_atom, dst, src)
    return dst
end
function _copyto!(copy_atom::AbstractCopyAtom, dst::NonOwningArray, src::NonOwningArray)
    @inline
    return _copyto!(copy_atom, dst, src, TrivialPred())
end
@generated function _copyto!(copy_atom::AbstractCopyAtom, dst::NonOwningArray{TD,N}, src::NonOwningArray{TS, N}, pred) where {TD, TS, N}
    expr = generate_copy_atom_loops(:sliced_dst, :sliced_src,  make_tuple(shape(layout(dst)[_1])), make_tuple(shape(layout(src)[_1])),
                                    num_val_src(copy_atom), num_val_dst(copy_atom))
    return quote
        dst_v = group_modes(dst, _2, StaticInt{$N}())
        src_v = group_modes(src, _2, StaticInt{$N}())
        @loopinfo unroll for i in 1:size(layout(src_v), 2)
            if pred(i)
                sliced_dst = view(dst_v, :, i)
                sliced_src = view(src_v, :, i)
                $expr
            end
        end
        return dst
    end
end

#Base.@assume_effects :foldable function _copyto!(copy_atom::AbstractCopyAtom, dst::NonOwningArray{TD, 1}, src::NonOwningArray{TS, 1}, pred) where {TD, TS}
#    @inline 
#    apply(copy_atom, dst, src)
#end
#Base.@assume_effects :foldable function _copyto!(copy_atom::AbstractCopyAtom, dst::NonOwningArray{TD, N}, src::NonOwningArray{TS, N}, pred) where {TD, TS, N}
#    src_v = group_modes(src, _2, StaticInt{N}())
#    dst_v = group_modes(dst, _2, StaticInt{N}())
#    for i in 1:size(src_v.layout, 2)
#        if pred(i)
#            apply(copy_atom, view(dst_v, :, i), view(src_v, :, i))
#        end
#    end
#end