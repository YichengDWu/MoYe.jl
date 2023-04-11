using CuTe, Test

@testset "Elementwise Comparison" begin
    @test elem_less((1,2,3), (1,2,4)) == true
    @test elem_less((1,2,3), (1,2,3)) == false
    @test elem_less((1,2,3), (1,2,3,1)) == true
    @test elem_less((1,2,3,1), (1,2,3)) == false

    @test elem_less(((1,2,3),), ((1,2,3),)) == false
    @test elem_less(((1,2,3),), ((1,2,4),)) == true
    @test elem_less(((1,2,3),), ((1,2,3),1)) == true
    @test elem_less(((1,2,3),1), ((1,2,3),2)) == true
    @test elem_less(((1,2,3),2), ((1,2,4),1)) == false

    @test elem_less(((1,2), (3,4)), ((5,6), (7,8))) == true
    @test elem_less(((1,2), (3,4)), ((1,2), (7,8))) == true
    @test elem_less(((1,2), (3,4)), ((1,2), (1,8))) == false
    @test elem_less(((1,2), (3,4)), ((3,4), (3,4))) == false
    @test elem_less(((1,2), (3,4)), ((3,4), (3,4,1))) == true
    @test elem_less(((1,2), (3,4)), ((3,4), (3,4),1)) == true
end

@testset "Colexicographical Comparison" begin
    @test colex_less((1,2,3), (1,2,4)) == true
    @test colex_less((1,2,3), (1,2,3)) == false
    @test colex_less((1,2,3), (1,1,2,3)) == true
    @test colex_less((1,1,2,3), (1,2,3)) == false

    @test colex_less(((1,2,3),), ((1,2,3),)) == false
    @test colex_less(((0,2,3),), ((1,2,3),)) == true
    @test colex_less(((1,2,3),), (1,(1,2,3))) == true
    @test colex_less((1,(1,2,3)), (2,(1,2,3))) == true
    @test colex_less((2, (1,2,3)), (1, (2,2,3))) == true

    @test colex_less(((1,2), (3,4)), ((5,6), (7,8))) == true
    @test colex_less(((1,2), (3,4)), ((1,2), (7,8))) == true
    @test colex_less(((1,2), (3,4)), ((1,2), (1,8))) == false
    @test colex_less(((1,2), (3,4)), ((3,4), (3,4))) == false
    @test colex_less(((1,2), (3,4)), ((3,4), (3,4,1))) == true
    @test colex_less(((1,2), (3,4)), ((3,4), (3,4),1)) == true
end
