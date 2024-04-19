"""
    fma!(::AbstractMMAOP, D, A, B, C)

Perform matrix multiply-and-accumulate computation, `A*B+C`, and store the result in D.
The available subtypes of `AbstractMMAOP`s are
```julia
"MMAOP_8x8x4_F64F64F64F64_TN" => "llvm.nvvm.mma.m8n8k4.row.col.f64"
"MMAOP_8x8x4_F32F16F16F16_TN" => "llvm.nvvm.mma.m8n8k4.row.col.f32.f16"
"MMAOP_8x8x4_F32F16F16F16_NT" => "llvm.nvvm.mma.m8n8k4.col.row.f32.f16"
"MMAOP_8x8x4_F32F16F16F16_TT" => "llvm.nvvm.mma.m8n8k4.col.col.f32.f16"
"MMAOP_8x8x4_F32F16F16F16_NN" => "llvm.nvvm.mma.m8n8k4.row.row.f32.f16"
"MMAOP_8x8x4_F32F16F16F32_TN" => "llvm.nvvm.mma.m8n8k4.row.col.f32.f32"
"MMAOP_8x8x4_F32F16F16F32_NT" => "llvm.nvvm.mma.m8n8k4.col.row.f32.f32"
"MMAOP_8x8x4_F32F16F16F32_TT" => "llvm.nvvm.mma.m8n8k4.col.col.f32.f32"
"MMAOP_8x8x4_F32F16F16F32_NN" => "llvm.nvvm.mma.m8n8k4.row.row.f32.f32"
"MMAOP_8x8x4_F16F16F16F16_TN" => "llvm.nvvm.mma.m8n8k4.row.col.f16.f16"
"MMAOP_8x8x4_F16F16F16F16_NT" => "llvm.nvvm.mma.m8n8k4.col.row.f16.f16"
"MMAOP_8x8x4_F16F16F16F16_TT" => "llvm.nvvm.mma.m8n8k4.col.col.f16.f16"
"MMAOP_8x8x4_F16F16F16F16_NN" => "llvm.nvvm.mma.m8n8k4.row.row.f16.f16"
"MMAOP_16x8x8_F16F16F16F16_TN" => "llvm.nvvm.mma.m16n8k8.row.col.f16.f16"
"MMAOP_16x8x16_F16F16F16F16_TN" => "llvm.nvvm.mma.m16n8k16.row.col.f16.f16"
"MMAOP_16x8x8_F32F16F16F32_TN" => "llvm.nvvm.mma.m16n8k8.row.col.f32.f32"
"MMAOP_16x8x16_F32F16F16F32_TN" => "llvm.nvvm.mma.m16n8k16.row.col.f32.f32"
"MMAOP_16x8x8_F32BF16BF16F32_TN" => "llvm.nvvm.mma.m16n8k8.row.col.bf16"
"MMAOP_16x8x16_F32BF16BF16F32_TN" => "llvm.nvvm.mma.m16n8k16.row.col.bf16"
"MMAOP_16x8x8_F32TF32TF32F32_TN" => "llvm.nvvm.mma.m16n8k8.row.col.tf32"
```

You can instantiate any of these `AbstractMMAOP`s and inspect the information about
the operation
```julia
julia> op =  MMAOP_16x8x8_F32TF32TF32F32_TN()
MMAOP_16x8x8_F32TF32TF32F32_TN()

julia> op.ARegisters        # Register type, and number of registers
MoYe.Registers{UInt32, 4}

julia> op.BRegisters
MoYe.Registers{UInt32, 2}

julia> op.CRegisters
MoYe.Registers{Float32, 4}
```

!!! note
    Do not use `mma` with `wmma.load` together. Their data layouts do not agree.
    The correct execution chain is ldmatrix + mma.
"""
function fma! end

# PTX types to LLVM types for registers
const ptx_to_llvm_reg = Dict(
    "f16"  => "<2 x half>",
    "f32"  => "float",
    "f64"  => "double",
    "s32"  => "i32",
    "b16"  => "i32",
    "s8"   => "i32",
    "u8"   => "i32",
    "s4"   => "i32",
    "u4"   => "i32",
    "b1"   => "i32",
    "bf16" => "i32",
    "tf32" => "i32",
)

# PTX types to julia types for registers
const ptx_to_jl_reg = Dict(
    "f16"  => NTuple{2, VecElement{Float16}},
    "f32"  => Float32,
    "f64"  => Float64,
    "s32"  => UInt32,
    "b16"  => UInt32,
    "s8"   => UInt32,
    "u8"   => UInt32,
    "s4"   => UInt32,
    "u4"   => UInt32,
    "b1"   => UInt32,
    "bf16" => UInt32,
    "tf32" => UInt32
)

const ptx_to_jl = Dict(
    "f16" => Float16,
    "f32" => Float32,
    "f64" => Float64,
    "s32"  => Int32,
    "s8" => Int8,
    "u8" => UInt8,
    "bf16" => BFloat16,
)

# geom to num of registers
const nregs = Dict(
    # u8/s8 -> s32 @ m16n16k16/m8n32k16/m32n8k16
    "m16n16k16:a:u8" => 2,
    "m16n16k16:a:s8" => 2,
    "m16n16k16:b:u8" => 2,
    "m16n16k16:b:s8" => 2,
    "m16n16k16:c:s32" => 8,
    "m16n16k16:d:s32" =>  8,
    "m8n32k16:a:u8" => 1,
    "m8n32k16:a:s8" =>  1,
    "m8n32k16:b:u8" =>  4,
    "m8n32k16:b:s8" =>  4,
    "m8n32k16:c:s32"=>8,
    "m8n32k16:d:s32"=>8,
    "m32n8k16:a:u8"=>4,
    "m32n8k16:a:s8"=>4,
    "m32n8k16:b:u8"=>1,
    "m32n8k16:b:s8"=>1,
    "m32n8k16:c:s32"=>8,
    "m32n8k16:d:s32"=>8,
    "m8n8k16:a:u8"=>1,
    "m8n8k16:a:s8"=>1,
    "m8n8k16:b:u8"=>1,
    "m8n8k16:b:s8"=>1,
    "m8n8k16:c:s32"=>2,
    "m8n8k16:d:s32"=>2,
    "m16n8k16:a:u8"=>2,
    "m16n8k16:a:s8"=>2,
    "m16n8k16:b:u8"=>1,
    "m16n8k16:b:s8"=>1,
    "m16n8k16:c:s32"=>4,
    "m16n8k16:d:s32"=>4,
    "m16n8k32:a:u8"=>4,
    "m16n8k32:a:s8"=>4,
    "m16n8k32:b:u8"=>2,
    "m16n8k32:b:s8"=>2,
    "m16n8k32:c:s32"=>4,
    "m16n8k32:d:s32"=>4,
    # u4/s4 -> s32 @ m8n8k32 (u4/s4)
    "m8n8k32:a:u4"=>1,
    "m8n8k32:a:s4"=>1,
    "m8n8k32:b:u4"=>1,
    "m8n8k32:b:s4"=>1,
    "m8n8k32:c:s32"=>2,
    "m8n8k32:d:s32"=>2,
    "m16n8k32:a:u4"=>2,
    "m16n8k32:a:s4"=>2,
    "m16n8k32:b:u4"=>1,
    "m16n8k32:b:s4"=>1,
    "m16n8k32:c:s32"=>4,
    "m16n8k32:d:s32"=>4,
    "m16n8k64:a:u4"=>4,
    "m16n8k64:a:s4"=>4,
    "m16n8k64:b:u4"=>2,
    "m16n8k64:b:s4"=>2,
    "m16n8k64:c:s32"=>4,
    "m16n8k64:d:s32"=>4,
    # b1 -> s32 @ m8n8k128(b1)
    "m8n8k128:a:b1"=>1,
    "m8n8k128:b:b1"=>1,
    "m8n8k128:c:s32"=>2,
    "m8n8k128:d:s32"=>2,
    "m16n8k128:a:b1"=>2,
    "m16n8k128:b:b1"=>1,
    "m16n8k128:c:s32"=>4,
    "m16n8k128:d:s32"=>4,
    "m16n8k256:a:b1"=>4,
    "m16n8k256:b:b1"=>2,
    "m16n8k256:c:s32"=>4,
    "m16n8k256:d:s32"=>4,
    # bf16 -> s32 @ m16n16k16/m8n32k16/m32n8k16
    "m16n16k16:a:bf16"=>4,
    "m16n16k16:b:bf16"=>4,
    "m8n32k16:a:bf16"=>2,
    "m8n32k16:b:bf16"=>8,
    "m32n8k16:a:bf16"=>8,
    "m32n8k16:b:bf16"=>2,
    "m16n8k16:a:bf16"=>4,
    "m16n8k16:b:bf16"=>2,
    "m16n8k16:c:f32"=>4,
    "m16n8k16:d:f32"=>4,
    "m16n8k8:a:bf16"=>2,
    "m16n8k8:b:bf16"=>1,
    "m16n8k8:c:f32"=>4,
    "m16n8k8:d:f32"=>4,
    "m8n8k4:a:f64"=>1,
    "m8n8k4:b:f64"=>1,
    "m8n8k4:c:f64"=>2,
    "m8n8k4:d:f64"=>2,
    # tf32 -> s32 @ m16n16k8
    "m16n16k8:a:tf32"=>4,
    "m16n16k8:b:tf32"=>4,
    "m16n8k4:a:tf32"=>2,
    "m16n8k4:b:tf32"=>1,
    "m16n8k4:c:f32"=>4,
    "m16n8k4:d:f32"=>4,
    "m16n8k8:a:tf32"=>4,
    "m16n8k8:b:tf32"=>2,
    "m16n8k8:c:f32"=>4,
    "m16n8k8:d:f32"=>4,
    "m8n8k4:a:f16"=>2,
    "m8n8k4:b:f16"=>2,
    "m16n8k8:a:f16"=>2,
    "m16n8k8:b:f16"=>1,
    "m16n8k8:c:f16"=>2,
    "m16n8k8:d:f16"=>2,
    "m16n8k8:c:f32"=>4,
    "m16n8k8:d:f32"=>4,
    "m16n8k16:a:f16"=>4,
    "m16n8k16:b:f16"=>2,
    "m16n8k16:c:f16"=>2,
    "m16n8k16:d:f16"=>2,
    "m16n8k16:c:f32"=>4,
    "m16n8k16:d:f32"=>4,
    # ldmatrix
    "m8n8:x1:b16"=>1,
    "m8n8:x2:b16"=>2,
    "m8n8:x4:b16"=>4,
)

# all other combinations have the smae size
const other_nregs = Dict(
    "a:f16" => 8,
    "b:f16" => 8,
    "c:f16" => 4,
    "d:f16" => 4,
    "c:f32" => 8,
    "d:f32" => 8,
)

function get_mmp_op_signature(alayout, blayout, satf, b1op)
    alayout = alayout == "row" ? "T" : "N"
    blayout = blayout == "row" ? "T" : "N"
    satf = isempty(satf) ? "" : "_SATURATE"
    if isempty(b1op)
        blop = ""
    elseif b1op == ".xor.popc"
        blop = "_XORPOPC"
    else
        blop = "_ANDPOPC"
    end
    return "$alayout$blayout$satf$blop"
end

function get_nregs(geom, frag, ptx_elt_type)
    return get(nregs, "$geom:$frag:$ptx_elt_type", get(other_nregs, "$frag:$ptx_elt_type", nothing))
end

# LLVMStruct lowers to {llvmT, llvmT, ...}, which matches the return type of the intrinsic
for N in unique(vcat(unique(values(nregs)), unique(values(other_nregs))))
    struct_ty = Symbol("LLVMStruct$N")

    @eval struct $struct_ty{T}
        Base.Cartesian.@nexprs $N i -> x_i::T
    end
    #@eval Base.convert(::Type{NTuple{$N, T}}, x::$struct_ty{T}) where {T} = ntuple(i -> getfield(x, i), $N)
end

function get_b1_ops(ptx_type)
    ptx_type != "b1" && return [""]
    return [".xor.popc", ".and.popc"]
end

function make_frag(geom, frag, ptx_elt_type)
   T, S = ptx_to_jl_reg[ptx_elt_type], get_nregs(geom, frag, ptx_elt_type)
   return Registers{T, S}
end

function mma_intrinsic_signature(ptx_type_d, ptx_type_a, ptx_type_b, ptx_type_c,)
    if ptx_type_a == "f16"
        return "$ptx_type_d.$ptx_type_c"         # FP16 ops identified by accumulator & result type.
    elseif ptx_type_a != ptx_type_b
        return "$ptx_type_a.$ptx_type_b"         # other ops are identified by input types.
    else
        return ptx_type_a                        # if input types are the same, it only appears once.
    end
end

function convert_geom(input_string::String)
    regex = r"(\d+)"
    matches = collect(eachmatch(regex, input_string))
    result = join((m.match for m in matches), "x")
    return result
end

function get_ccall_args(ARegisters, BRegisters, CRegisters, DRegisters)
    a_frag_ty = eltype(ARegisters)
    b_frag_ty = eltype(BRegisters)
    c_frag_ty = eltype(CRegisters)
    d_frag_ty = eltype(DRegisters)

    a_sz = length(ARegisters)
    b_sz = length(BRegisters)
    c_sz = length(CRegisters)
    d_sz = length(DRegisters)

    a_types = ntuple(i -> a_frag_ty, a_sz)
    b_types = ntuple(i -> b_frag_ty, b_sz)
    c_types = ntuple(i -> c_frag_ty, c_sz)
    d_types = @eval $(Symbol(:LLVMStruct,d_sz)){$d_frag_ty}

    a_vars = ntuple(i -> :(a[$i]), a_sz)
    b_vars = ntuple(i -> :(b[$i]), b_sz)
    c_vars = ntuple(i -> :(c[$i]), c_sz)

    return  a_types, b_types, c_types, d_types, a_vars, b_vars, c_vars, d_frag_ty, d_sz
end


function is_mma_variant_supported(type_a, type_b, type_c, type_d, geom, layout_a, layout_b, satf)
    if !(isempty(satf) || (type_a in ["s8", "u8", "s4", "u4"]))
        return false
    end

    if geom == "m8n8k4" && (type_c == "f32") && (type_d != "f32")
        return false
    end

    if geom == "m16n8k8" && (type_a != type_b || type_c != type_d)
        return false
    end

    if geom == "m16n8k16" && (type_c != type_d)
        return false
    end

    if !((geom == "m8n8k4") && type_a == "f16")
        return layout_a == "row" && layout_b == "col"
    end
    return true
end

function make_mma_ops(geoms, types_a, types_b, types_c, types_d)
    struct_names = []
    for (geom, type_a, type_c) in Iterators.product(geoms,  types_a, types_c)
        for (type_b, type_d) in Iterators.product(ifelse(isempty(types_b), [type_a], types_b),
                                                  ifelse(isempty(types_d), [type_c], types_d))

            for (alayout, blayout, satf) in Iterators.product(["row", "col"], ["row", "col"], ["", ".satfinite"])
                if !is_mma_variant_supported(type_a, type_b, type_c, type_d, geom, alayout, blayout, satf)
                    continue
                end

                for b1op in get_b1_ops(type_a)
                    # Step 1: Construct the MMA_OP struct
                    struct_name = "MMAOP_$(convert_geom(geom))_$(uppercase(type_d*type_a*type_b*type_c))_$(get_mmp_op_signature(alayout, blayout, satf, b1op))"

                    DRegisters = make_frag(geom, "d", type_d)
                    ARegisters = make_frag(geom, "a", type_a)
                    BRegisters = make_frag(geom, "b", type_b)
                    CRegisters = make_frag(geom, "c", type_c)

                    _struct_name = Symbol(struct_name)
                    @eval struct $_struct_name <: AbstractMMAOP{$DRegisters, $ARegisters, $BRegisters, $CRegisters} end
                    @eval export $_struct_name

                    # Step 2: Construct the intrinsic 
                    intrinsic_signature = mma_intrinsic_signature(type_d, type_a, type_b, type_c)
                    mma_intrinsic = "llvm.nvvm.mma$b1op.$geom.$alayout.$blayout$satf.$intrinsic_signature"

                    push!(struct_names, struct_name => mma_intrinsic)
                    a_types, b_types, c_types, d_types, a_vars, b_vars, c_vars, d_frag_ty, d_sz = get_ccall_args(ARegisters(), BRegisters(), CRegisters(), DRegisters())

                    if d_sz == 1
                        @eval @inline function (::$_struct_name)(a, b, c)
                            return ccall($mma_intrinsic, llvmcall, $d_frag_ty, ($(a_types...), $(b_types...), $(c_types...)), $(a_vars...), $(b_vars...), $(c_vars...))
                        end

                        @eval @inline function fma!(op::$_struct_name, d, a, b, c)
                            val = op(a,b,c)
                            return unsafe_store!(pointer(d), val, 1)
                        end
                    else
                        @eval @inline function (::$_struct_name)(a, b, c)
                            return ccall($mma_intrinsic, llvmcall, $d_types, ($(a_types...), $(b_types...), $(c_types...)), $(a_vars...), $(b_vars...), $(c_vars...))
                        end

                        @eval @inline function fma!(op::$_struct_name, d, a, b, c)
                            val = op(a,b,c)
                            ptr = pointer(d)
                            Base.Cartesian.@nexprs $d_sz i -> unsafe_store!(ptr, getfield(val, i), i)
                            return d
                        end
                    end
                end
            end
        end
    end
    return struct_names
end

function get_mma_ops()
    vcat(
    # 8x8x4
    make_mma_ops(["m8n8k4"], ["f64"], [], ["f64"], []), # 1
    make_mma_ops(["m16n8k4", "m16n8k8"], ["tf32"], [], ["f32"], []), # 1 
    make_mma_ops(["m16n8k16", "m16n8k8"], ["bf16"], [], ["f32"], []), # 2
    make_mma_ops(["m8n8k4", "m16n8k8", "m16n8k16"], ["f16"], [],[ "f16", "f32"], ["f16", "f32"]), # 16
    make_mma_ops(["m8n8k16", "m16n8k16", "m16n8k32"], ["s8", "u8"], ["s8", "u8"], ["s32"], []), # 24 
    make_mma_ops(["m8n8k32", "m16n8k32", "m16n8k64"], ["s4", "u4"], ["s4", "u4"], ["s32"], []), # 24
    make_mma_ops(["m8n8k128", "m16n8k128", "m16n8k256"], ["b1"], [], ["s32"], []), # 6 
    )
end

get_mma_ops()