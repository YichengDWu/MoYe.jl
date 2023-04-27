using Shambles, Test

function test_alloc()
    slayout = @Layout (2, 3)
    x = CuTeArray{Float32}(undef, slayout)
    fill!(x, 1.0f0)
    return sum(x)
end

@test @allocated(test_alloc()) == 0

@testset "Constructors" begin
    @test_nowarn CuTeArray{Float32}(undef, static((2, 3)))
    @test_throws MethodError CuTeArray{Float32}(undef, (2, 3))

    @test_nowarn CuTeArray{Float32}(undef, static((2, 3)), GenRowMajor)
    @test_nowarn CuTeArray{Float32}(undef, static((2, 3)), GenColMajor)

    A = rand(3)
    ca = CuTeArray(pointer(A), static((3, 1)))
    ca2 = CuTeArray(pointer(A), static((3, 1)), GenRowMajor)
    @test ca.engine isa ViewEngine
    @test ca2.engine isa ViewEngine
end

@testset "Array Operations" begin
    @testset "View" begin
        ca = CuTeArray{Float32}(undef, static((2, 3)))
        va = view(ca, :, 1)
        @test va isa CuTeArray
        @test va.engine isa ViewEngine
        @test va.layout.shape == tuple(static(2))

        va2 = view(ca, :, :)
        @test va2 isa CuTeArray
        @test va2.engine isa ViewEngine
        @test va2.layout.shape == tuple(static(2), static(3))
    end

    @testset "Copy" begin
        ca = CuTeArray{Float32}(undef, static((2, 3)))
        ca2 = copy(ca)
        @test ca2 isa CuTeArray
        @test ca2.engine isa ArrayEngine
        @test ca2 == ca

        A = ones(6)
        ca3 = CuTeArray(pointer(A), static(6))
        ca4 = copy(ca3)
        @test ca4 isa CuTeArray
        @test ca4.engine isa ArrayEngine
        @test ca4 == ca3
    end

    @testset "similar" begin
        ca = CuTeArray{Float32}(undef, static((2, 3)))
        ca2 = similar(ca)
        @test ca2 isa CuTeArray
        @test ca2.engine isa ArrayEngine
        @test ca2.layout == ca.layout

        A = ones(6)
        ca3 = CuTeArray(pointer(A), static(6))
        ca4 = similar(ca3)
        @test ca4 isa CuTeArray
        @test ca4.engine isa ArrayEngine
        @test ca4.layout == ca3.layout
    end
end

@testset "BLAS" begin
    @testset "fill! and sum" begin
        ca = CuTeArray{Float32}(undef, static((2, 3)))
        fill!(ca, 1.0f0)
        @test all(ca .== 1.0f0)
        @test sum(ca) == 6.0f0
    end
end
