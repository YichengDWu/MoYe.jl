abstract type AbstractLdMatrix{SRegisters, DRegisters} <: AbstractCopyOperation{SRegisters, DRegisters} end

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
LD_U32x4_N()

julia> ans.DRegisters
Registers{UInt32, 4}
```
!!! note
    These intrinsics have bugs in LLVM 14, but was fixed in LLVM 15.
"""
function copyto!(ldmatrix::AbstractLdMatrix, dest::MoYeArray{UInt32}, src::MoYeArray{UInt128}) end

function get_ld_type(dest_sz, layout)
    signature = layout == "" ? "N" : "T"
    e_type = signature == "N" ? "U32" : "U16"
    sz = signature == "N" ? dest_sz : 2dest_sz
    ld_type = "LDSM_$(e_type)x$(sz)_$signature"
    return ld_type
end

function get_ldmatrix_ops()
    ptr_type = LLVMPtr{UInt32, AS.Shared}
    src_type, src_sz = UInt128, 1 # each thread provides a 128 bits pointer
    dest_type = UInt32

    ld_ops = []
    for (dest_sz, layout) in Iterators.product([1, 2, 4], ["", ".trans"])
        ld_type = get_ld_type(dest_sz, layout)
        @eval struct $(Symbol(ld_type)) <: AbstractLdMatrix{Registers{$src_type, $src_sz}, Registers{$dest_type, $dest_sz}} end
        #@eval export $(Symbol(ld_type))

        intrinsic = "llvm.nvvm.ldmatrix.sync.aligned.m8n8.x$(dest_sz)$layout.b16"
        push!(ld_ops, ld_type => intrinsic)

        llvm_struct = Symbol("LLVMStruct$dest_sz")
        ret_type = @eval $llvm_struct{$dest_type}
        if isone(dest_sz)
            @eval @inline function (::$(Symbol(ld_type)))(src_addr::$ptr_type)
                return ccall($intrinsic, llvmcall, $dest_type, ($ptr_type,), src_addr)
            end

            @eval function Base.copyto!(op::$(Symbol(ld_type)), dest::LocalArray{$dest_type}, src::SharedArray{$src_type})
                src_ptr = pointer(src)
                val = op(recast(UInt32, src_ptr))
                return unsafe_store!(dest, val, 1)
            end
        else
            @eval @inline function (::$(Symbol(ld_type)))(src_addr::$ptr_type)
                return ccall($intrinsic, llvmcall, $ret_type, ($ptr_type,), src_addr)
            end

            @eval function Base.copyto!(op::$(Symbol(ld_type)), dest::LocalArray{$dest_type}, src::SharedArray{$src_type})
                src_ptr = pointer(src)
                val = op(recast(UInt32, src_ptr))
                dest_ptr = pointer(dest)
                Base.Cartesian.@nexprs $dest_sz i -> unsafe_store!(dest_ptr, getfield(val, i), i)
                return dest
            end
        end
    end
    return ld_ops
end

get_ldmatrix_ops()
