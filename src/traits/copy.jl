abstract type AbstractCopyTraits <: AbstractTraits end
struct CopyTraits{C <: AbstractCPOP, TS, TD, LT, LS, LD, LR} <: AbstractCopyTraits
    threadid::LT
    srclayout::LS
    dstlayout::LD
    reflayout::LR
end

function CopyTraits{OP}(threadid, srclayout, dstlayout, reflayout=srclayout) where {S, D, OP<:AbstractCPOP{S,D}}
    return CopyTraits{OP, S, D, typeof(threadid), typeof(srclayout), typeof(dstlayout), typeof(reflayout)}(threadid, srclayout, dstlayout, reflayout)
end

function CopyTraits{CPOP_UNIVERSAL{S,D}}() where {S,D}
    threadid = make_layout(One())  # 1 thread per operation
    srclayout = make_layout((One(), static(sizeof(S) * 8))) # thr -> bit
    dstlayout = make_layout((One(), static(sizeof(D) * 8)))
    return CopyTraits{CPOP_UNIVERSAL{S,D}}(threadid, srclayout, dstlayout)
end
