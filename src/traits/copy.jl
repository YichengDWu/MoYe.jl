abstract type AbstractCopyTraits{OP} <: AbstractTraits end

struct CopyTraits{C <: AbstractCopyOperation, LT, LS, LD, LR} <: AbstractCopyTraits{C}
    copy_op::C
    threadid::LT
    srclayout::LS
    dstlayout::LD
    reflayout::LR
end

function CopyTraits{OP}(threadid, srclayout, dstlayout, reflayout=srclayout) where {OP<:AbstractCopyOperation}
    copy_op = OP()
    return CopyTraits{typeof(copy_op), typeof(threadid), typeof(srclayout), typeof(dstlayout), typeof(reflayout)}(copy_op, threadid, srclayout, dstlayout, reflayout)
end

function CopyTraits{UniversalCopy{S,D}}() where {S,D}
    threadid = make_layout(One())  # 1 thread per operation
    srclayout = make_layout((One(), static(sizeof(S) * 8))) # thr -> bit
    dstlayout = make_layout((One(), static(sizeof(D) * 8)))
    return CopyTraits{UniversalCopy{S,D}}(threadid, srclayout, dstlayout)
end

function copyto_unpack!(::AbstractCopyTraits{OP}, dst::MoYeArray, src::MoYeArray) where {OP}
    cpop = OP()
    registers_src = cpop.SRegisters
    registers_dst = cpop.DRegisters

    regtype_src = eltype(registers_src)
    regtype_dst = eltype(registers_dst)
    regnum_src = length(registers_src)
    regnum_dst = length(registers_dst)

    rs = recast(regtype_src, src)
    rd = recast(regtype_dst, dst)

    @assert size(rs.layout) == StaticInt{regnum_src}()
    @assert size(rd.layout) == StaticInt{regnum_dst}()
    copyto!(cpop, rd, rs)
    return dst
end
