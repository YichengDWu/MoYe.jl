abstract type CPOP{SRegisters, DRegisters} end

@inline apply(copy_op::CPOP, dest::LLVMPtr, src::LLVMPtr) = copy_op(dest, src)

struct CPOP_UNIVERSAL{TS, TD} <: CPOP{Registers{TS, 1}, Registers{TD, 1}} end

@inline CPOP_UNIVERSAL{S}() where {S} = CPOP_UNIVERSAL{S,S}()

function (::CPOP_UNIVERSAL{TS, TD})(dest::LLVMPtr{TD}, src::LLVMPtr{TS}) where {TS, TD}
    @inline
    align_src = Base.datatype_alignment(TS)
    align_dst = Base.datatype_alignment(TD)

    return unsafe_store!(dest, unsafe_load(src, 1, Val(align_src)), 1, Val(align_dst))
end
