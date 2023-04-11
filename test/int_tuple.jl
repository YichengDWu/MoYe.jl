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
