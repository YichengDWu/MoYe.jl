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
    "tf32" => "i32"
)

const ptx_reg_pattern = Dict(
    "f16" => r"%hh[0-9]+",
    "f32" => r"%f[0-9]+",
    "f64" => r"%fd[0-9]+",
)

get_mma_type(ptx_type::String) = ptx_to_llvm[ptx_type], get(ptx_reg_pattern, ptx_type, r"%[a-z]+[0-9]+")
