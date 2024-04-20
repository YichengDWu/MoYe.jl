function _get_layouts(::Tuple{StaticInt{16}, StaticInt{8}, StaticInt{8}}, AElType, CElType)                    
    A = @Layout ((4, 8), (2, 2)) ((32, 1), (16, 8))
    B = @Layout ((4, 8), 2) ((16, 1), 8)
    C = A
    return A, B, C
end

function _get_layouts(::Tuple{StaticInt{16}, StaticInt{8}, StaticInt{16}},
                      AElType::Type{<:AbstractFloat},
                      CElType::Type{<:AbstractFloat})
    A = @Layout ((4, 8), (2, 2, 2)) ((32, 1), (16, 8, 128))
    B = @Layout ((4, 8), (2, 2)) ((16, 1), (8, 64))
    C = @Layout ((4, 8), (2, 2)) ((32, 1), (16, 8))
    return A, B, C
end

function _get_layouts(::Tuple{StaticInt{16}, StaticInt{8}, StaticInt{16}}, AElType, CElType)
    A = @Layout ((4, 8), (4, 2, 2)) ((64,1), (16, 8, 256))
    B = @Layout ((4, 8), (4, 2)) ((32, 1), (8, 128))
    C = @Layout ((4, 8), (2, 2)) ((32, 1), (16, 8))
    return A, B, C
end

function _get_layouts(::Tuple{StaticInt{8}, StaticInt{8}, StaticInt{4}}, AElType, CElType)
    A = @Layout ((4, 8), 1) ((8, 1), 0)
    B = A
    C = @Layout ((4, 8), 2) ((16, 1), 8)
    return A, B, C
end

function _get_layouts(::Tuple{StaticInt{8}, StaticInt{8}, StaticInt{16}}, AElType, CElType)
    A = @Layout ((4, 8), 4) ((32, 1), 8)
    B = A
    C = @Layout ((4, 8), 2) ((16, 1), 8)
    return A, B, C
end

function _get_layouts(::Tuple{StaticInt{16}, StaticInt{8}, StaticInt{32}}, AElType, CElType)
    A = @Layout ((4, 8), (4, 2, 2)) ((64, 1), (16, 8, 256))
    B = @Layout ((4, 8), (4, 2)) ((32, 1), (8, 128))
    C = @Layout ((4, 8), (2, 2)) ((32, 1), (16, 8))
    return A, B, C
end


function make_sm80_mmatraits(mmaops)
    for mmaop in mmaops
        @eval @inline thr_id(::MMATraits{$mmaop}) = @Layout(32)

        mmop_ins = @eval $mmaop()
        A, B, C = _get_layouts(shape_mnk(mmop_ins), valtype_a(mmop_ins), valtype_c(mmop_ins))

        @eval layout_a(::MMATraits{$mmaop}) = $A
        @eval layout_b(::MMATraits{$mmaop}) = $B
        @eval layout_c(::MMATraits{$mmaop}) = $C
    end
end

# 16x8x16
make_sm80_mmatraits((
    # 16x8x8
    :MMAOP_16x8x8_F16F16F16F16_TN,
    # :MMAOP_16x8x8_F32TF32TF32F32_TN,
    :MMAOP_16x8x8_F32BF16BF16F32_TN,

    # 16x8x16
    :MMAOP_16x8x16_F16F16F16F16_TN,
    :MMAOP_16x8x16_F32F16F16F32_TN,
    :MMAOP_16x8x16_F32BF16BF16F32_TN,
    :MMAOP_16x8x16_S32S8S8S32_TN,
    :MMAOP_16x8x16_S32S8S8S32_TN_SATURATE,
    :MMAOP_16x8x16_S32S8U8S32_TN,
    :MMAOP_16x8x16_S32S8U8S32_TN_SATURATE,
    :MMAOP_16x8x16_S32U8S8S32_TN,
    :MMAOP_16x8x16_S32U8S8S32_TN_SATURATE,
    :MMAOP_16x8x16_S32U8U8S32_TN,
    :MMAOP_16x8x16_S32U8U8S32_TN_SATURATE,

    # 8x8x4 TODO： add complex types
    :MMAOP_8x8x4_F64F64F64F64_TN,

    # 8x8x16
    #:MMAOP_8x8x16_S32S8S8S32_TN,
    :MMAOP_8x8x16_S32S8S8S32_TN_SATURATE,
    :MMAOP_8x8x16_S32S8U8S32_TN,
    :MMAOP_8x8x16_S32S8U8S32_TN_SATURATE,
    :MMAOP_8x8x16_S32U8S8S32_TN,
    :MMAOP_8x8x16_S32U8S8S32_TN_SATURATE,
    :MMAOP_8x8x16_S32U8U8S32_TN,
    :MMAOP_8x8x16_S32U8U8S32_TN_SATURATE,

    # 16x8x32
    :MMAOP_16x8x32_S32S8S8S32_TN,
    :MMAOP_16x8x32_S32S8S8S32_TN_SATURATE,
    :MMAOP_16x8x32_S32S8U8S32_TN,
    :MMAOP_16x8x32_S32S8U8S32_TN_SATURATE,
    :MMAOP_16x8x32_S32U8S8S32_TN,
    :MMAOP_16x8x32_S32U8S8S32_TN_SATURATE,
    :MMAOP_16x8x32_S32U8U8S32_TN,
    :MMAOP_16x8x32_S32U8U8S32_TN_SATURATE,
))


# special cases
@inline thr_id(::MMATraits{MMAOP_16x8x8_F32TF32TF32F32_TN}) = @Layout(32)
@inline layout_a(::MMATraits{MMAOP_16x8x8_F32TF32TF32F32_TN}) = @Layout ((4, 8), (2, 2)) ((16, 1), (8, 64))
@inline layout_b(::MMATraits{MMAOP_16x8x8_F32TF32TF32F32_TN}) = @Layout ((4, 8), 2) ((8, 1), 32)
@inline layout_c(::MMATraits{MMAOP_16x8x8_F32TF32TF32F32_TN}) = @Layout ((4, 8), (2, 2)) ((32, 1), (16, 6))

@inline thr_id(::MMATraits{MMAOP_16x8x4_F32TF32TF32F32_TN}) = @Layout(32)
@inline layout_a(::MMATraits{MMAOP_16x8x4_F32TF32TF32F32_TN}) = @Layout ((4, 8), 1) ((8, 1),08)
@inline layout_b(::MMATraits{MMAOP_16x8x4_F32TF32TF32F32_TN}) = @Layout ((4, 8), (2, 2)) ((32, 1), (16, 8))

@inline thr_id(::MMATraits{MMAOP_16x8x256_S32B1B1S32_TN_XORPOPC}) = @Layout(32)
@inline layout_a(::MMATraits{MMAOP_16x8x256_S32B1B1S32_TN_XORPOPC}) = @Layout (32, (8, 4, 2, 2)) (64, (64, 16 ,8, 2048))
@inline layout_b(::MMATraits{MMAOP_16x8x256_S32B1B1S32_TN_XORPOPC}) = @Layout (32, (32, 2)) (64, (1, 1024))