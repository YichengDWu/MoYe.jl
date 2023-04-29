using MoYe, Test, JET

@testset "Capacity" begin
    @test capacity(((2, 3, (1, 1)), 4)) == 24
    @test_opt capacity(((2, 3, (1, 1)), 4))
end

@testset "Shape Division" begin
    @test MoYe.shape_div((12, 3), (2, 3)) == (6, 1)
    @test_opt MoYe.shape_div((12, 3), (2, 3))

    @test MoYe.shape_div((12, 3), 3) == (4, 3)
    @test_opt MoYe.shape_div((12, 3), 3)

    @test MoYe.shape_div(12, (3, 4)) == 1
    @test_opt MoYe.shape_div(12, (3, 4))
end

@testset "Slice" begin
    @test MoYe.slice((3, 4), (2, :)) == (4,)
    @test_opt MoYe.slice((3, 4), (2, :))

    @test MoYe.slice((3, (4, 5)), (:, (2, :))) == (3, 5)
    @test_opt MoYe.slice((3, (4, 5)), (:, (2, :)))

    @test MoYe.slice(((2, 4), (4, 2)), (:, (:, :))) == ((2, 4), 4, 2)
    @test_opt MoYe.slice(((2, 4), (4, 2)), (:, (:, :)))
end

@testset "Elementwise Comparison" begin
    @test elem_less((1, 2, 3), (1, 2, 4)) == true
    @test elem_less((1, 2, 3), (1, 2, 3)) == false
    @test elem_less((1, 2, 3), (1, 2, 3, 1)) == true
    @test elem_less((1, 2, 3, 1), (1, 2, 3)) == false

    @test elem_less(((1, 2, 3),), ((1, 2, 3),)) == false
    @test elem_less(((1, 2, 3),), ((1, 2, 4),)) == true
    @test elem_less(((1, 2, 3),), ((1, 2, 3), 1)) == true
    @test elem_less(((1, 2, 3), 1), ((1, 2, 3), 2)) == true
    @test elem_less(((1, 2, 3), 2), ((1, 2, 4), 1)) == false

    @test elem_less(((1, 2), (3, 4)), ((5, 6), (7, 8))) == true
    @test elem_less(((1, 2), (3, 4)), ((1, 2), (7, 8))) == true
    @test elem_less(((1, 2), (3, 4)), ((1, 2), (1, 8))) == false
    @test elem_less(((1, 2), (3, 4)), ((3, 4), (3, 4))) == false
    @test elem_less(((1, 2), (3, 4)), ((3, 4), (3, 4, 1))) == true
    @test elem_less(((1, 2), (3, 4)), ((3, 4), (3, 4), 1)) == true

    @test_opt elem_less(((1, 2), (3, 4)), ((3, 4), (3, 4), 1))
end

@testset "Colexicographical Comparison" begin
    @test colex_less((1, 2, 3), (1, 2, 4)) == true
    @test colex_less((1, 2, 3), (1, 2, 3)) == false
    @test colex_less((1, 2, 3), (1, 1, 2, 3)) == true
    @test colex_less((1, 1, 2, 3), (1, 2, 3)) == false

    @test colex_less(((1, 2, 3),), ((1, 2, 3),)) == false
    @test colex_less(((0, 2, 3),), ((1, 2, 3),)) == true
    @test colex_less(((1, 2, 3),), (1, (1, 2, 3))) == true
    @test colex_less((1, (1, 2, 3)), (2, (1, 2, 3))) == true
    @test colex_less((2, (1, 2, 3)), (1, (2, 2, 3))) == true

    @test colex_less(((1, 2), (3, 4)), ((5, 6), (7, 8))) == true
    @test colex_less(((1, 2), (3, 4)), ((2, 3), (3, 4))) == true
    @test colex_less(((1, 2), (3, 4)), ((1, 3), (3, 4))) == true
    @test colex_less(((5, 4), (3, 4)), ((3, 4), (3, 4))) == false
    @test colex_less(((1, 2), (3, 4)), ((1, 1, 2), (3, 4))) == true

    @test_opt colex_less(((1, 2), (3, 4)), ((1, 1, 2), (3, 4)))
end

@testset "Increment" begin
    @test increment((1, 1), (3, 4)) == (2, 1)
    @test increment((3, 1), (3, 4)) == (1, 2)
    @test increment((3, 4), (3, 4)) == (1, 1)

    @test increment((2, (2, 1), 1), (2, (2, 3), 3)) == (1, (1, 2), 1)
    @test increment((2, (2, 2), 1), (2, (2, 3), 3)) == (1, (1, 3), 1)
    @test increment((2, (2, 3), 1), (2, (2, 3), 3)) == (1, (1, 1), 2)
    @test increment((2, (2, 3), 3), (2, (2, 3), 3)) == (1, (1, 1), 1)

    @test_opt increment((2, (2, 3), 3), (2, (2, 3), 3))
end
