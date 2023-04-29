using MoYe, Test, CUDA
using Static: One

if CUDA.functional()
    @testset "Global Memory" begin
        a = CUDA.rand(Float32, 10)
        a = CUDA.cudaconvert(a)
        MoYeArray(pointer(a), static((2, 5)))
    end

    @testset "Shared Memory" begin
        ptr = MoYe.SharedMemory(Float32, static(10))
        MoYeArray(ptr, static((2, 5)))
    end

    @testset "Register Memory" begin
        @test_nowarn MoYeArray{Float32}(undef, static((2, 5)))

        a = CUDA.rand(Float32, 8, 16)
        a = CUDA.cudaconvert(a)
        gmem_8sx16d = MoYeArray(pointer(a), (static(8), 16))
        rmem = make_fragment_like(view(gmem_8sx16d, :, 1))
        @test rmem.layout.shape == tuple(static(8))
        @test rmem.layout.stride == tuple(One())
        @test length(rmem.engine) == 8
    end
end
