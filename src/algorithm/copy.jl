struct TrivialMask end
@inline Base.getindex(::TrivialMask, i) = true

@inline function maksed_copyto!(dest::CuTeArray, src::CuTeArray, mask)
    copy_op = select_elementwise_copy(src, dest)
    for i in eachindex(dest)
        if mask[i]
            apply(copy_op, pointer(dest, i), pointer(src, i))
        end
    end
end

@inline function copyto_vec!(dest::CuTeArray{TD}, src::CuTeArray{TS}, ::Type{TV}) where {TD,TS,TV}
    b = ManualMemory.preserve_buffer(dest)
    b2 = ManualMemory.preserve_buffer(src)

    GC.@preserve b b2 begin
        if (sizeof(TD) == sizeof(TS)) && sizeof(TV) > sizeof(TD)
            src_v = recast(TV, src)
            dest_v = recast(TV, dest)
            maksed_copyto!(dest_v, src_v, TrivialMask())
        else
            maksed_copyto!(dest, src, TrivialMask())
        end
    end
end
