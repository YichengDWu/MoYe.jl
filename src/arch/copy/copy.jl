const CP_SYNC_ENABLED = true # TODO: make this configurable. >=SM_80

abstract type CPOP{SRegisters, DRegisters} end

struct CPOP_UNIVERSAL{TS, TD} <: CPOP{Registers{TS, 1}, Registers{TD, 1}}
    @inline CPOP_UNIVERSAL{S}() where {S} = new{S,S}()
end

function _unsafe_copyto!(dest::LLVMPtr{TD}, src::LLVMPtr{TS}, ::CPOP_UNIVERSAL{TS, TD}) where {TS, TD}
    @inline
    unsafe_copyto!(dest, src, 1)
end
