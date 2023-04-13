using CuTe, Test

@testset "Coalesce" begin
    @test coalesce(make_layout((2,(1,6)), (1,(6,2)))) == make_layout(12,1)
end
