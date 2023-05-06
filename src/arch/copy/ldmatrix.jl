abstract type AbstractLdMatrix{SRegisters, DRegisters} <: AbstractCPOP{SRegisters, DRegisters} end

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
    load(::AbstractLdMatrix, src_addr::LLVMPtr) where {T}

Load one or multiple matrices from shared memory to registers. The available `LdMatrix`s are:

```julia
# Type => LLVM intrinsic
"LDSM_U32x1_N" => "llvm.nvvm.ldmatrix.sync.aligned.m8n8.x1.b16"
"LDSM_U32x2_N" => "llvm.nvvm.ldmatrix.sync.aligned.m8n8.x2.b16"
"LDSM_U32x4_N" => "llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16"
"LDSM_U16x2_T" => "llvm.nvvm.ldmatrix.sync.aligned.m8n8.x1.trans.b16"
"LDSM_U16x4_T" => "llvm.nvvm.ldmatrix.sync.aligned.m8n8.x2.trans.b16"
"LDSM_U16x8_T" => "llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.trans.b16"
```
You can inspect how many registers are used to store the matrix per thread by
```julia
julia> LDSM_U32x4_N()
LD_U32x4_N()

julia> ans.DRegisters
Registers{UInt32, 4}
```
!!! note
    Would not work with LLVM 14
"""
function load end

@inline load(op::AbstractLdMatrix, src_addr::LLVMPtr) = op(src_addr)

function get_ld_type(d_sz, layout)
    signature = layout == "" ? "N" : "T"
    e_type = signature == "N" ? "U32" : "U16"
    sz = signature == "N" ? d_sz : 2d_sz
    ld_type = "LDSM_$(e_type)x$(sz)_$signature"
    return ld_type
end

function get_ldmatrix_ops()
    ptr_type = LLVMPtr{UInt32, AS.Shared}
    s_type, s_sz = UInt128, 1 # each thread provides a 128 bits pointer
    dst_type = UInt32

    ld_ops = []
    for (d_sz, layout) in Iterators.product([1, 2, 4], ["", ".trans"])
        ld_type = get_ld_type(d_sz, layout)
        @eval struct $(Symbol(ld_type)) <: AbstractLdMatrix{Registers{$s_type, $s_sz}, Registers{$dst_type, $d_sz}} end
        #@eval export $(Symbol(ld_type))

        intrinsic = "llvm.nvvm.ldmatrix.sync.aligned.m8n8.x$(d_sz)$layout.b16"
        push!(ld_ops, ld_type => intrinsic)

        llvm_struct = Symbol("LLVMStruct$d_sz")
        ret_type = @eval $llvm_struct{$dst_type}
        if isone(d_sz)
            @eval @inline function (::$(Symbol(ld_type)))(src_addr::LLVMPtr)
                _src_addr = $LLVM.Interop.addrspacecast($ptr_type, src_addr)
                return tuple(ccall($intrinsic, llvmcall, $dst_type, ($ptr_type,), _src_addr))
            end
        else
            @eval @inline function (::$(Symbol(ld_type)))(src_addr::LLVMPtr)
                _src_addr = $LLVM.Interop.addrspacecast($ptr_type, src_addr)
                return convert(NTuple{$d_sz, $dst_type}, ccall($intrinsic, llvmcall, $ret_type, ($ptr_type,), _src_addr))
            end
        end
    end
    return ld_ops
end

get_ldmatrix_ops()
