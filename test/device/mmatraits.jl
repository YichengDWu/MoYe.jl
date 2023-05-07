using Test, MoYe

@testset "Size of MMATraits" begin
    for mmaop in [MoYe.MMAOP_16x8x8_F16F16F16F16_TN,
        MoYe.MMAOP_16x8x8_F32F16F16F32_TN,
        MoYe.MMAOP_16x8x8_F32BF16BF16F32_TN,
        MoYe.MMAOP_16x8x16_F16F16F16F16_TN,
        MoYe.MMAOP_16x8x16_F32F16F16F32_TN,
        MoYe.MMAOP_16x8x16_F32BF16BF16F32_TN]
        @test sizeof(MoYe.MMATraits{mmaop}()) == 0
    end
end
