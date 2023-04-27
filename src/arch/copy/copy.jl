const CP_SYNC_ENABLED = true # TODO: make this configurable. >=SM_80

abstract type CPOP{SRegisters, DRegisters} end

@inline apply(copy_op::CPOP, dest::LLVMPtr, src::LLVMPtr) = copy_op(dest, src)

struct CPOP_UNIVERSAL{TS, TD} <: CPOP{Registers{TS, 1}, Registers{TD, 1}}
    @inline CPOP_UNIVERSAL{S}() where {S} = new{S,S}()
end

function (::CPOP_UNIVERSAL{TS, TD})(dest::LLVMPtr{TD}, src::LLVMPtr{TS}) where {TS, TD}
    @inline
    unsafe_copyto!(dest, src, 1)
end
