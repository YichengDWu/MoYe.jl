using CuTe, Test

@testset "Flatten" begin
    flatten(make_layout(((4,3), 1), ((3, 1), 0))) == make_layout((4, 3, 1), (3, 1, 0))
end
@testset "Coalesce" begin
    @test coalesce(make_layout((2, (1, 6)), (1, (6, 2)))) == make_layout(12, 1)
end

@testset "Composition" begin
    @test make_layout(20, 2) ∘ make_layout((4, 5), (1, 4)) == make_layout((4, 5), (2, 8))
    @test make_layout(20, 2) ∘ make_layout((4, 5), (5, 1)) == make_layout((4, 5), (10, 2))
end

@testset "Complement" begin
    @test complement(make_layout(4,1), 24) == make_layout(6, 4)
    @test complement(make_layout(6,4), 24) == make_layout(4, 1)
end
