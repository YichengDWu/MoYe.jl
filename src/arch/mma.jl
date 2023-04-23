abstract type MMAOP{DRegisters, ARegisters, BRegisters, CRegisters} end
struct Registers{T,S} end


"""
    mma(::MMAOP, A, B, C)

Perform matrix multiply-and-accumulate computation, `A*B+C`. The available `MMAOP`s are
```julia
 "MMA_8x8x4_F64F64F64F64_TN"
 "MMA_8x8x4_F32F16F16F16_TN"
 "MMA_8x8x4_F32F16F16F16_NT"
 "MMA_8x8x4_F32F16F16F16_TT"
 "MMA_8x8x4_F32F16F16F16_NN"
 "MMA_8x8x4_F32F16F16F32_TN"
 "MMA_8x8x4_F32F16F16F32_NT"
 "MMA_8x8x4_F32F16F16F32_TT"
 "MMA_8x8x4_F32F16F16F32_NN"
 "MMA_8x8x4_F16F16F16F16_TN"
 "MMA_8x8x4_F16F16F16F16_NT"
 "MMA_8x8x4_F16F16F16F16_TT"
 "MMA_8x8x4_F16F16F16F16_NN"
 "MMA_16x8x8_F16F16F16F16_TN"
 "MMA_16x8x16_F16F16F16F16_TN"
 "MMA_16x8x8_F32F16F16F32_TN"
 "MMA_16x8x16_F32F16F16F32_TN"
 "MMA_16x8x8_F32BF16BF16F32_TN"
 "MMA_16x8x16_F32BF16BF16F32_TN"
 "MMA_16x8x8_F32TF32TF32F32_TN"
```
You can instantiate any of these `MMAOP`s and inspect the information about the operation

```julia
julia> op =  MMA_16x8x8_F32TF32TF32F32_TN()
MMA_16x8x8_F32TF32TF32F32_TN()

julia> op.ARegisters        # Register type, and number of registers
CuTe.Registers{UInt32, 4}

julia> op.BRegisters
CuTe.Registers{UInt32, 2}

julia> op.CRegisters
CuTe.Registers{Float32, 4}
```
"""
function mma end

@inline Base.eltype(::Registers{T}) where {T} = T
@inline Base.length(::Registers{T, L}) where {T, L} = L

function Base.getproperty(obj::MMAOP{DRegisters, ARegisters,
                                     BRegisters, CRegisters},
                          sym::Symbol) where {DRegisters, ARegisters,
                                              BRegisters, CRegisters}
    if sym === :DRegisters
        return DRegisters
    elseif sym === :ARegisters
        return ARegisters
    elseif sym === :BRegisters
        return BRegisters
    elseif sym === :CRegisters
        return CRegisters
    else
        return getfield(obj,sym)
    end
end

# PTX types to LLVM types
const ptx_to_llvm = Dict(
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

# PTX types to julia types
const ptx_to_jl = Dict(
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

const ptx_reg_pattern = Dict(
    "f16" => r"%hh[0-9]+",
    "f32" => r"%f[0-9]+",
    "f64" => r"%fd[0-9]+",
)

#function get_mma_type(ptx_type::String)
#    return ptx_to_llvm[ptx_type]#, get(ptx_reg_pattern, ptx_type, r"%r[0-9]+")
#end

const nregs = Dict(
    "m16n16k16:a:u8" => 2,
    "m16n16k16:a:s8" => 2,
    "m16n16k16:b:u8" => 2,
    "m16n16k16:b:s8" => 2,
    "m16n16k16:c:s32" => 8,
    "m16n16k16:d:s32" => 8,

    "m8n32k16:a:u8" => 1,
    "m8n32k16:a:s8" => 1,
    "m8n32k16:b:u8" => 4,
    "m8n32k16:b:s8" => 4,
    "m8n32k16:c:s32" => 8,
    "m8n32k16:d:s32" => 8,

    "m32n8k16:a:u8" => 4,
    "m32n8k16:a:s8" => 4,
    "m32n8k16:b:u8" => 1,
    "m32n8k16:b:s8" => 1,
    "m32n8k16:c:s32" => 8,
    "m32n8k16:d:s32" => 8,

    "m8n8k16:a:u8" => 1,
    "m8n8k16:a:s8" => 1,
    "m8n8k16:b:u8" => 1,
    "m8n8k16:b:s8" => 1,
    "m8n8k16:c:s32" => 2,
    "m8n8k16:d:s32" => 2,

    "m16n8k16:a:u8" => 2,
    "m16n8k16:a:s8" => 2,
    "m16n8k16:b:u8" => 1,
    "m16n8k16:b:s8" => 1,
    "m16n8k16:c:s32" => 4,
    "m16n8k16:d:s32" => 4,

    "m16n8k32:a:u8" => 4,
    "m16n8k32:a:s8" => 4,
    "m16n8k32:b:u8" => 2,
    "m16n8k32:b:s8" => 2,
    "m16n8k32:c:s32" => 4,
    "m16n8k32:d:s32" => 4,

    # u4/s4 -> s32 @ m8n8k32 (u4/s4)
    "m8n8k32:a:u4" => 1,
    "m8n8k32:a:s4" => 1,
    "m8n8k32:b:u4" => 1,
    "m8n8k32:b:s4" => 1,
    "m8n8k32:c:s32" => 2,
    "m8n8k32:d:s32" => 2,

    "m16n8k32:a:u4" => 2,
    "m16n8k32:a:s4" => 2,
    "m16n8k32:b:u4" => 1,
    "m16n8k32:b:s4" => 1,
    "m16n8k32:c:s32" => 4,
    "m16n8k32:d:s32" => 4,

    "m16n8k64:a:u4" => 4,
    "m16n8k64:a:s4" => 4,
    "m16n8k64:b:u4" => 2,
    "m16n8k64:b:s4" => 2,
    "m16n8k64:c:s32" => 4,
    "m16n8k64:d:s32" => 4,

    # b1 -> s32 @ m8n8k128(b1)
    "m8n8k128:a:b1" => 1,
    "m8n8k128:b:b1" => 1,
    "m8n8k128:c:s32" => 2,
    "m8n8k128:d:s32" => 2,

    "m16n8k128:a:b1" => 2,
    "m16n8k128:b:b1" => 1,
    "m16n8k128:c:s32" => 4,
    "m16n8k128:d:s32" => 4,

    "m16n8k256:a:b1" => 4,
    "m16n8k256:b:b1" => 2,
    "m16n8k256:c:s32" => 4,
    "m16n8k256:d:s32" => 4,

    "m16n16k16:a:bf16" =>  4,
    "m16n16k16:b:bf16" =>  4,
    "m8n32k16:a:bf16"  =>  2,
    "m8n32k16:b:bf16"  =>  8,
    "m32n8k16:a:bf16"  =>  8,
    "m32n8k16:b:bf16"  =>  2,

    "m16n8k16:a:bf16" =>  4,
    "m16n8k16:b:bf16" =>  2,
    "m16n8k16:c:f32"  =>  4,
    "m16n8k16:d:f32"  =>  4,
    "m16n8k8:a:bf16"  =>  2,
    "m16n8k8:b:bf16"  =>  1,
    "m16n8k8:c:f32"   =>  4,
    "m16n8k8:d:f32"   =>  4,

    "m8n8k4:a:f64" =>  1,
    "m8n8k4:b:f64" =>  1,
    "m8n8k4:c:f64" =>  2,
    "m8n8k4:d:f64" =>  2,

    "m16n16k8:a:tf32" =>  4,
    "m16n16k8:b:tf32" =>  4,

    "m16n8k4:a:tf32" =>  2,
    "m16n8k4:b:tf32" =>  1,
    "m16n8k4:c:f32"  =>  4,
    "m16n8k4:d:f32"  =>  4,
    "m16n8k8:a:tf32" =>  4,
    "m16n8k8:b:tf32" =>  2,
    "m16n8k8:c:f32"  =>  4,
    "m16n8k8:d:f32"  =>  4,

    "m8n8k4:a:f16"   =>  2,
    "m8n8k4:b:f16"   =>  2,
    "m16n8k8:a:f16"  =>  2,
    "m16n8k8:b:f16"  =>  1,
    "m16n8k8:c:f16"  =>  2,
    "m16n8k8:d:f16"  =>  2,
    "m16n8k8:c:f32"  =>  4,
    "m16n8k8:d:f32"  =>  4,
    "m16n8k16:a:f16" =>  4,
    "m16n8k16:b:f16" =>  2,
    "m16n8k16:c:f16" =>  2,
    "m16n8k16:d:f16" =>  2,
    "m16n8k16:c:f32" =>  4,
    "m16n8k16:d:f32" =>  4,

    # ldmatrix
    "m8n8:x1:b16" =>  1,
    "m8n8:x2:b16" =>  2,
    "m8n8:x4:b16" =>  4,
)

const other_nregs = Dict(
    "a:f16" => 8,
    "b:f16" => 8,
    "c:f16" => 4,
    "d:f16" => 4,
    "c:f32" => 8,
    "d:f32" => 8,
)

const op_signature_to_layout = Dict(
    "TN" => "row.col",
    "NT" => "col.row",
    "TT" => "col.col",
    "NN" => "row.row",
)

for N in unique(vcat(unique(values(nregs)), unique(values(other_nregs))))
    struct_ty = Symbol("LLVMStruct$N")

    @eval struct $struct_ty{T}
        Base.Cartesian.@nexprs $N i -> x_i::T
    end

    @eval Base.convert(::Type{NTuple{$N, T}}, x::$struct_ty{T}) where {T} = ntuple(i -> getfield(x, i), $N)
end

function get_nregs(geom, frag, ptx_elt_type)
    return get(nregs, "$geom:$frag:$ptx_elt_type", get(other_nregs, "$frag:$ptx_elt_type", nothing))
end

function make_frag(geom, frag, ptx_elt_type)
   T, S = ptx_to_jl[ptx_elt_type], get_nregs(geom, frag, ptx_elt_type)
   if S === nothing
    @show T, geom, frag, S, ptx_elt_type
    @show "$geom:$frag:$ptx_elt_type"
   end
   return Registers{T, S}
end

function get_mmaop_name(prefix, geom, type_d, type_a, type_b, type_c, signature)
    return "$(prefix)_$(geom)_$(type_d)$(type_a)$(type_b)$(type_c)_$(signature)"
end

function mma_signature(ptx_type_d, ptx_type_a, ptx_type_b, ptx_type_c,)
    if ptx_type_a == "f16"
        return "$ptx_type_d.$ptx_type_c"
    elseif ptx_type_a != ptx_type_b
        return "$ptx_type_a.$ptx_type_b"
    else
        return ptx_type_a
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

function make_mma_ops(geoms, types_a, types_b, types_c, types_d, signatures)
    struct_names = String[]
    for (geom, type_a, type_c) in Iterators.product(geoms,  types_a, types_c)
        for (type_b, type_d) in Iterators.product(ifelse(isempty(types_b), [type_a], types_b),
                                                ifelse(isempty(types_d), [type_c], types_d))
            for signature in signatures
                struct_name = "MMA_$(convert_geom(geom))_$(uppercase(type_d*type_a*type_b*type_c))_$(signature)"
                push!(struct_names, struct_name)

                DRegisters = make_frag(geom, "d", type_d)
                ARegisters = make_frag(geom, "a", type_a)
                BRegisters = make_frag(geom, "b", type_b)
                CRegisters = make_frag(geom, "c", type_c)

                _struct_name = Symbol(struct_name)
                @eval struct $_struct_name <: MMAOP{$DRegisters, $ARegisters, $BRegisters, $CRegisters} end
                @eval export $_struct_name

                intrinsic_signature = mma_signature(type_d, type_a, type_b, type_c)
                layout = op_signature_to_layout[signature]
                mma_intrinsic = "llvm.nvvm.mma.$geom.$layout.$intrinsic_signature"

                a_types, b_types, c_types, d_types, a_vars, b_vars, c_vars, d_frag_ty, d_sz = get_ccall_args(ARegisters(), BRegisters(), CRegisters(), DRegisters())
                @eval @inline mma(::$_struct_name, a, b, c) = convert(NTuple{$d_sz, $d_frag_ty}, ccall($mma_intrinsic, llvmcall, $d_types, ($(a_types...), $(b_types...), $(c_types...)), $(a_vars...), $(b_vars...), $(c_vars...)))
            end
        end
    end
    return struct_names
end

function get_mma_ops()
    vcat(
    # 8x8x4
    make_mma_ops(["m8n8k4"], ["f64"], [], ["f64"], [], ["TN"]),
    make_mma_ops(["m8n8k4"], ["f16"], [], ["f16", "f32"], ["f32"],  ["TN", "NT", "TT", "NN"]),
    make_mma_ops(["m8n8k4"], ["f16"], [], ["f16"], [], ["TN", "NT", "TT", "NN"]),

    # 16x8x8 16x8x16
    make_mma_ops(["m16n8k8", "m16n8k16"], ["f16"], [], ["f16"], [],  ["TN"]),
    make_mma_ops(["m16n8k8", "m16n8k16"], ["f16"], [], ["f32"], [],  ["TN"]),
    make_mma_ops(["m16n8k8", "m16n8k16"], ["bf16"], [], ["f32"], [],  ["TN"]),
    make_mma_ops(["m16n8k8"], ["tf32"], [], ["f32"], [],  ["TN"]),


    #make_mma_ops(["m8n8k16", "m16n8k16", "m16n8k32"], ["s8", "u8"], ["s8", "u8"], ["s32"], [], ["TN"])
    #make_mma_ops(["m8n8k32", "m16n8k32", "m16n8k64"], ["s4", "u4"], ["s4", "u4"], ["s32"], [], ["TN"]),
    #make_mma_ops(["m8n8k128", "m16n8k128", "m16n8k256"], ["b1"], [], ["s32"], [], ["TN"])
    )
end

get_mma_ops()
export mma

#= Currently generated MMA OP
 "MMA_8x8x4_F64F64F64F64_TN"
 "MMA_8x8x4_F32F16F16F16_TN"
 "MMA_8x8x4_F32F16F16F16_NT"
 "MMA_8x8x4_F32F16F16F16_TT"
 "MMA_8x8x4_F32F16F16F16_NN"
 "MMA_8x8x4_F32F16F16F32_TN"
 "MMA_8x8x4_F32F16F16F32_NT"
 "MMA_8x8x4_F32F16F16F32_TT"
 "MMA_8x8x4_F32F16F16F32_NN"
 "MMA_8x8x4_F16F16F16F16_TN"
 "MMA_8x8x4_F16F16F16F16_NT"
 "MMA_8x8x4_F16F16F16F16_TT"
 "MMA_8x8x4_F16F16F16F16_NN"
 "MMA_16x8x8_F16F16F16F16_TN"
 "MMA_16x8x16_F16F16F16F16_TN"
 "MMA_16x8x8_F32F16F16F32_TN"
 "MMA_16x8x16_F32F16F16F32_TN"
 "MMA_16x8x8_F32BF16BF16F32_TN"
 "MMA_16x8x16_F32BF16BF16F32_TN"
 "MMA_16x8x8_F32TF32TF32F32_TN"
 =#

#function make_ldmatrix_ops(geoms, frags, types)
#    for (geom, frag, ptx_type) in Iterators.product(geoms, frags, types)
#        make_frag(geom, frag, ptx_type)
#    end
#end

#make_ldmatrix_ops(["m8n8"], ["x1", "x2", "x4"], ["b16"])

const space_map = Dict(
    ".global" => AS.Global,
    ".shared" => AS.Shared,
    ".const"  => AS.Constant,
    ".local"  => AS.Local,
    ".param"  => 101,
    ""        => AS.Generic,
    ".generic" => AS.Generic,
)
