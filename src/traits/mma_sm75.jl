@inline thr_id(::MMATraits{MMAOP_16x8x8_F32F16F16F32_TN}) = @Layout(32)
@inline layout_a(::MMATraits{MMAOP_16x8x8_F32F16F16F32_TN}) = @Layout ((4, 8), (2,2)) ((32, 1), (16, 8))
@inline layout_b(::MMATraits{MMAOP_16x8x8_F32F16F16F32_TN}) = @Layout ((4, 8), 2) ((16, 1), 8)
@inline layout_c(::MMATraits{MMAOP_16x8x8_F32F16F16F32_TN}) = @Layout ((4, 8), (2, 2)) ((32, 1), (16, 8))

@inline thr_id(::MMATraits{MMAOP_8x8x16_S32S8S8S32_TN}) = @Layout(32)
@inline layout_a(::MMATraits{MMAOP_8x8x16_S32S8S8S32_TN}) = @Layout ((4, 8), 4) ((32, 1), 8)
@inline layout_b(::MMATraits{MMAOP_8x8x16_S32S8S8S32_TN}) = @Layout ((4, 8), 2) ((32, 1), 8)
@inline layout_c(::MMATraits{MMAOP_8x8x16_S32S8S8S32_TN}) = @Layout ((4, 8), 2) ((16, 1), 8)
