@inline function collective_copyto!(tiled_copy, dest, src)
    return quote
        thr_copy = get_slice($(esc(tiled_copy)), Int(threadIdx().x))
        thr_D = partition_D(thr_copy, $(esc(dest)))
        thr_S = partition_S(thr_copy, $(esc(src)))

        copyto!($(esc(tiled_copy)), thr_D, thr_S)
    end
end

"""
    @collective tiled_xxx f(args...)

Thread block level collective operation.
"""
macro collective(tiled_copy, ex)
    @capture(ex, f_(args__)) || error("unexpected expression")
    if f == :copyto!
        @capture(args__, (dest_, src_)) || error("unexpected arguments for copyto!")
        return collective_copyto!(tiled_copy, dest_, src_)
    end
end
