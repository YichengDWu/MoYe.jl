using CuTe, Test

function test_alloc()
    slayout = @Layout (2,3)
    x = CuTeArray{Float32}(undef, slayout)
    fill!(x, 1.0f0)
    sum(x)
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
