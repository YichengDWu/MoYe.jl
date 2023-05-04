struct CopyTraits{C <: CPOP, TS, TD, LT, LS, LD, LR}
    threadid::LT
    srclayout::LS
    dstlayout::LD
    reflayout::LR
end

function CopyTraits{OP}(threadid, srclayout, dstlayout, reflayout=srclayout) where {S, D, OP<:CPOP{S,D}}
    return CopyTraits{OP, S, D, typeof(threadid), typeof(srclayout), typeof(dstlayout), typeof(reflayout)}(threadid, srclayout, dstlayout, reflayout)
end

function CopyTraits{CPOP_UNIVERSAL{S,D}}() where {S,D}
    threadid = make_layout(One())  # 1 thread per operation
    srclayout = make_layout((One(), static(sizeof(S) * 8))) # thr -> bit
    dstlayout = make_layout((One(), static(sizeof(D) * 8)))
    return CopyTraits{CPOP_UNIVERSAL{S,D}}(threadid, srclayout, dstlayout)
end

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
    return CopyTraits{CPOP_ASYNC_CACHEALWAYS{S,D}}(threadid, srclayout, dstlayout, reflayout)
end

function select_elementwise_copy(src::MoYeArray{TS}, dest::MoYeArray{TD}) where {TS, TD}
    @static if CP_SYNC_ENABLED
        if isgmem(src) && issmem(dest) && sizeof(TS) == sizeof(TD)
            return CPOP_ASYNC_CACHEALWAYS{TS,TD}()
        else
            return CPOP_UNIVERSAL{TS,TD}()
        end
    else
        return CPOP_UNIVERSAL{TS,TD}()
    end
end
