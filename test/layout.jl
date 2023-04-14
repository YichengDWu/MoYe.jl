using CuTe, Test

@testset "Coalesce" begin
    @test coalesce(make_layout((2, (1, 6)), (1, (6, 2)))) == make_layout(12, 1)
end

@testset "Composition" begin
    @test make_layout(20, 2) ∘ make_layout((4, 5), (1, 4)) == make_layout((4, 5), (2, 8))
    @test make_layout(20, 2) ∘ make_layout((4, 5), (5, 1)) == make_layout((4, 5), (10, 2))
end
