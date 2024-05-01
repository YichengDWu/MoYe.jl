abstract type AbstractLdMatrix{SRegisters, DRegisters} <: AbstractCopyOp{SRegisters, DRegisters} end

function Base.getproperty(obj::AbstractLdMatrix{SRegisters, DRegisters},
                          sym::Symbol) where {SRegisters, DRegisters}
    if sym === :DRegisters
        return DRegisters
    elseif sym === :SRegisters
        return SRegisters
    else
        return getfield(obj,sym)
    end
end

function Base.propertynames(::AbstractLdMatrix)
    return (:SRegisters, :DRegisters)
end

struct LDSM_U32x1_N <: AbstractLdMatrix{Registers{UInt128, 1}, Registers{UInt32, 1}} end
struct LDSM_U32x2_N <: AbstractLdMatrix{Registers{UInt128, 1}, Registers{UInt32, 2}} end
struct LDSM_U32x4_N <: AbstractLdMatrix{Registers{UInt128, 1}, Registers{UInt32, 4}} end

struct LDSM_U16x2_T <: AbstractLdMatrix{Registers{UInt128, 1}, Registers{UInt32, 1}} end
struct LDSM_U16x4_T <: AbstractLdMatrix{Registers{UInt128, 1}, Registers{UInt32, 2}} end
struct LDSM_U16x8_T <: AbstractLdMatrix{Registers{UInt128, 1}, Registers{UInt32, 4}} end

@inline function (::LDSM_U32x1_N)(src_addr::LLVMPtr{UInt32, AS.Shared})
    return ccall("llvm.nvvm.ldmatrix.sync.aligned.m8n8.x1.b16", llvmcall, UInt32, (LLVMPtr{UInt32, AS.Shared},), src_addr)
end

@inline function (::LDSM_U32x2_N)(src_addr::LLVMPtr{UInt32, AS.Shared})
    return ccall("llvm.nvvm.ldmatrix.sync.aligned.m8n8.x2.b16", llvmcall, LLVMStruct2{UInt32}, (LLVMPtr{UInt32, AS.Shared},), src_addr)
end

@inline function (::LDSM_U32x4_N)(src_addr::LLVMPtr{UInt32, AS.Shared})
    return ccall("llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16", llvmcall, LLVMStruct4{UInt32}, (LLVMPtr{UInt32, AS.Shared},), src_addr)
end

@inline function (::LDSM_U16x2_T)(src_addr::LLVMPtr{UInt32, AS.Shared})
    return ccall("llvm.nvvm.ldmatrix.sync.aligned.m8n8.x1.trans.b16", llvmcall, UInt32, (LLVMPtr{UInt32, AS.Shared},), src_addr)
end

@inline function (::LDSM_U16x4_T)(src_addr::LLVMPtr{UInt32, AS.Shared})
    return ccall("llvm.nvvm.ldmatrix.sync.aligned.m8n8.x2.trans.b16", llvmcall, LLVMStruct2{UInt32}, (LLVMPtr{UInt32, AS.Shared},), src_addr)
end

@inline function (::LDSM_U16x8_T)(src_addr::LLVMPtr{UInt32, AS.Shared})
    return ccall("llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.trans.b16", llvmcall, LLVMStruct4{UInt32}, (LLVMPtr{UInt32, AS.Shared},), src_addr)
end

function Base.copyto!(op::LDSM_U32x1_N, dest::LocalArray{UInt32}, src::SharedArray{UInt128})
    @inline 
    src_ptr = pointer(src)
    val = op(recast(UInt32, src_ptr))
    return unsafe_store!(pointer(dest), val, 1)
end

@inbounds function Base.copyto!(op::LDSM_U32x2_N, dest::LocalArray{UInt32}, src::SharedArray{UInt128})
    @inline 
    src_ptr = pointer(src)
    val = op(recast(UInt32, src_ptr))
    Base.Cartesian.@nexprs 2 i -> dest[i] = getfield(val, i)
    return dest
end

@inbounds function Base.copyto!(op::LDSM_U32x4_N, dest::LocalArray{UInt32}, src::SharedArray{UInt128})
    @inline 
    src_ptr = pointer(src)
    val = op(recast(UInt32, src_ptr))
    Base.Cartesian.@nexprs 4 i -> dest[i] = getfield(val, i)
    return dest
end

function Base.copyto!(op::LDSM_U16x2_T, dest::LocalArray{UInt32}, src::SharedArray{UInt128})
    @inline 
    src_ptr = pointer(src)
    val = op(recast(UInt32, src_ptr))
    return unsafe_store!(pointer(dest), val, 1)
end

@inbounds function Base.copyto!(op::LDSM_U16x4_T, dest::LocalArray{UInt32}, src::SharedArray{UInt128})
    @inline 
    src_ptr = pointer(src)
    val = op(recast(UInt32, src_ptr))
    dest_ptr = pointer(dest)
    Base.Cartesian.@nexprs 2 i -> dest[i] = getfield(val, i)
    return dest
end

@inbounds function Base.copyto!(op::LDSM_U16x8_T, dest::LocalArray{UInt32}, src::SharedArray{UInt128})
    @inline 
    src_ptr = pointer(src)
    val = op(recast(UInt32, src_ptr))
    dest_ptr = pointer(dest)
    Base.Cartesian.@nexprs 4 i -> dest[i] = getfield(val, i)
    return dest
end

"""
    copyto!(ldmatrix::AbstractLdMatrix, dest::MoYeArray{UInt32}, src::MoYeArray{UInt128})

Load data from shared memory to registers. The available `AbstractLdMatrix`s are:

```julia
# Type => LLVM intrinsic
"LDSM_U32x1_N" => "llvm.nvvm.ldmatrix.sync.aligned.m8n8.x1.b16"
"LDSM_U32x2_N" => "llvm.nvvm.ldmatrix.sync.aligned.m8n8.x2.b16"
"LDSM_U32x4_N" => "llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16"
"LDSM_U16x2_T" => "llvm.nvvm.ldmatrix.sync.aligned.m8n8.x1.trans.b16"
"LDSM_U16x4_T" => "llvm.nvvm.ldmatrix.sync.aligned.m8n8.x2.trans.b16"
"LDSM_U16x8_T" => "llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.trans.b16"
```
You can inspect the number and the type of  registers used per thread by
```julia
julia> LDSM_U32x4_N()
LDSM_U32x4_N()

julia> ans.DRegisters
Registers{UInt32, 4}
```
"""
function Base.copyto!(ldmatrix::AbstractLdMatrix, dest::MoYeArray, src::MoYeArray)
    throw(MethodError(copyto!, (ldmatrix, dest, src)))
end


const ldmatrix_ops_list = [
    "LDSM_U32x1_N" => "llvm.nvvm.ldmatrix.sync.aligned.m8n8.x1.b16"
"LDSM_U32x2_N" => "llvm.nvvm.ldmatrix.sync.aligned.m8n8.x2.b16"
"LDSM_U32x4_N" => "llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16"
"LDSM_U16x2_T" => "llvm.nvvm.ldmatrix.sync.aligned.m8n8.x1.trans.b16"
"LDSM_U16x4_T" => "llvm.nvvm.ldmatrix.sync.aligned.m8n8.x2.trans.b16"
"LDSM_U16x8_T" => "llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.trans.b16"
]

export LDSM_U32x1_N, LDSM_U32x2_N, LDSM_U32x4_N, LDSM_U16x2_T, LDSM_U16x4_T, LDSM_U16x8_T