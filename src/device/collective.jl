@inline function collective_copyto!(tiled_copy, dest, src)
    return quote
        thr_copy = get_thread_slice($(esc(tiled_copy)), Int(threadIdx().x))
        thr_D = partition_D(thr_copy, $(esc(dest)))
        thr_S = partition_S(thr_copy, $(esc(src)))

        copyto!($(esc(tiled_copy)), thr_D, thr_S)
    end
end


macro collective(ex)
    @capture(ex, f_(tiled_copy_, dest_, src_)) || error("unexpected expression")
    if f == :copyto!
        return collective_copyto!(tiled_copy, dest, src)
    end
end
