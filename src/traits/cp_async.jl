
function CopyTraits{CPOP_ASYNC_CACHEALWAYS{S,D}}() where {S,D}
    threadid = make_layout(One()) # 1 thread per operation
    srclayout = make_layout((One(), static(sizeof(S)*8)))
    dstlayout = make_layout((One(), static(sizeof(D)*8)))
    return CopyTraits{CPOP_ASYNC_CACHEALWAYS{S,D}}(threadid, srclayout, dstlayout)
end

function CopyTraits{CPOP_ASYNC_CACHEGLOBAL{S,D}}() where {S,D}
    threadid = make_layout(One()) # 1 thread per operation
    srclayout = make_layout((One(), static(sizeof(S)*8)))
    dstlayout = make_layout((One(), static(sizeof(D)*8)))
    return CopyTraits{CPOP_ASYNC_CACHEALWAYS{S,D}}(threadid, srclayout, dstlayout)
end

function select_elementwise_copy(src::MoYeArray{TS}, dest::MoYeArray{TD}) where {TS, TD}
    @static if CP_SYNC_ENABLED
        if isgmem(src) && issmem(dest) && sizeof(TS) == sizeof(TD)
            return CPOP_ASYNC_CACHEALWAYS{TS,TD}()
        else
            return UniversalCopy{TS,TD}()
        end
    else
        return UniversalCopy{TS,TD}()
    end
end
