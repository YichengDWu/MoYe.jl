using Test, Shambles

@testset "Size of MMATraits" begin
    for mmaop in [MMAOP_16x8x8_F16F16F16F16_TN,
                 MMAOP_16x8x8_F32F16F16F32_TN,
                 MMAOP_16x8x8_F32BF16BF16F32_TN,
                 MMAOP_16x8x16_F16F16F16F16_TN,
                 MMAOP_16x8x16_F32F16F16F32_TN,
                 MMAOP_16x8x16_F32BF16BF16F32_TN]
        @test sizeof(MMATraits{mmaop}()) == 0
    end
end
