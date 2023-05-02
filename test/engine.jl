using MoYe, Test

function test_alloc()
    x = ArrayEngine{Float32}(one, static(10))
    @gc_preserve sum(x)
end

@test @allocated(test_alloc()) == 0

@testset "Const" begin
    x = ArrayEngine{Float32}(undef, static(10))
    cx = ConstViewEngine(x)
    @test cx isa ConstViewEngine
    @test_nowarn cx[1]
    @test_throws MethodError cx[1] = 1.0f0
end
