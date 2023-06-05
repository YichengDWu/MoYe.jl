abstract type AbstractCopyOperation{SRegisters, DRegisters} <: PTXOperation end

function Base.getproperty(obj::AbstractCopyOperation{SRegisters, DRegisters},
                          sym::Symbol) where {SRegisters, DRegisters}
    if sym === :DRegisters
        return DRegisters
    elseif sym === :SRegisters
        return SRegisters
    else
        return getfield(obj,sym)
    end
end

function Base.propertynames(::AbstractCopyOperation)
    return (:SRegisters, :DRegisters)
end

@inline Adapt.adapt(to, x::AbstractCopyOperation) = x

struct UniversalCopy{TS, TD} <: AbstractCopyOperation{Registers{TS, 1}, Registers{TD, 1}} end

@inline UniversalCopy{S}() where {S} = UniversalCopy{S,S}()

function (::UniversalCopy{TS, TD})(dest::LLVMPtr{TD}, src::LLVMPtr{TS}) where {TS, TD}
    @inline
    align_src = Base.datatype_alignment(TS)
    align_dst = Base.datatype_alignment(TD)

    return unsafe_store!(dest, unsafe_load(src, 1, Val(align_src)), 1, Val(align_dst))
end

# on cpu
function (::UniversalCopy{TS, TD})(dest::Ptr{TD}, src::Ptr{TS}) where {TS, TD}
    @inline
    return unsafe_store!(dest, unsafe_load(src, 1), 1)
end

function Base.copyto!(op::UniversalCopy, dest::MoYeArray, src::MoYeArray)
    op(pointer(dest), pointer(src))
    return dest
end
