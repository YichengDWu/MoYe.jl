abstract type AbstractCopyOp{SRegisters, DRegisters} <: PTXOperation end

function Base.getproperty(obj::AbstractCopyOp{SRegisters, DRegisters},
                          sym::Symbol) where {SRegisters, DRegisters}
    if sym === :DRegisters
        return DRegisters
    elseif sym === :SRegisters
        return SRegisters
    else
        return getfield(obj,sym)
    end
end

function Base.propertynames(::AbstractCopyOp)
    return (:SRegisters, :DRegisters)
end

# default implementation, 1 value per thread
function Base.copyto!(op::AbstractCopyOp, dest::MoYeArray, src::MoYeArray)
    op(pointer(dest), pointer(src))
    return dest
end

@inline Adapt.adapt(to, x::AbstractCopyOp) = x

struct UniversalCopy{TS, TD} <: AbstractCopyOp{Registers{TS, 1}, Registers{TD, 1}} end

@inline UniversalCopy{S}() where {S} = UniversalCopy{S,S}()

function (::UniversalCopy{TS, TD})(dest::LLVMPtr, src::LLVMPtr) where {TS, TD}
    @inline
    src = recast(TS, src)
    dest = recast(TD, dest)
    align_src = Base.datatype_alignment(TS)
    align_dst = Base.datatype_alignment(TD)

    return unsafe_store!(dest, unsafe_load(src, 1, Val(align_src)), 1, Val(align_dst))
end

# the following methods should be moved if LocalArray has an address space
function (::UniversalCopy{TS, TD})(dest::Ptr, src::Ptr) where {TS, TD}
    @inline
    src = recast(TS, src)
    dest = recast(TD, dest)
    return unsafe_store!(dest, unsafe_load(src))
end
function (::UniversalCopy{TS, TD})(dest::Ptr, src::LLVMPtr) where {TS, TD}
    @inline
    src = recast(TS, src)
    dest = recast(TD, dest)
    return unsafe_store!(dest, unsafe_load(src, 1, Val(Base.datatype_alignment(TS))))
end
function (::UniversalCopy{TS, TD})(dest::LLVMPtr{TD}, src::Ptr{TS}) where {TS, TD}
    @inline
    src = recast(TS, src)
    dest = recast(TD, dest)
    return unsafe_store!(dest, unsafe_load(src), 1, Val(Base.datatype_alignment(TD)))
end
