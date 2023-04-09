using Test, CuTe

@testset "Constant" begin
    @testset "Constructor" begin
        @test_nowarn Constant{Int, 1}(1)
        @test_throws MethodError Constant{Int, 1.0}(1)
        @test Constant{Int}(Val(1)) == Constant{Int, 1}(1)
        @test Constant{Float32}(Val(1)) == Constant{Float32, 1}(1.0f0)
    end

    @test convert(Int, Constant{Bool, true}(true)) == 1
    @test Base.eltype(Constant{Int, 1}(1)) == Int
end
