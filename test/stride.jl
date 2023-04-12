using CuTe, Test

@testset "Coordinate to Index" begin
    @testset "1-D Coordinate" begin
        @test coord_to_index(5, (3,4), (2,8)) == 11
        @test coord_to_index(7, (3,4), (2,8)) == 17
        @test coord_to_index(12, (3,4), (2,8)) == 29
    end

    @testset "h-D Coordinate" begin
        @test coord_to_index((2,2), (3,4), (2,8)) == 11
        @test coord_to_index((1,3), (3,4), (2,8)) == 17
        @test coord_to_index((3,4), (3,4), (2,8)) == 29
    end

    @tesetset "R-D Coordinate" begin
        @test coord_to_index((2,3), (3, (2,2)), (2, (16, 8))) == 11
        @test coord_to_index((1,2), (3, (2,2)), (2, (16, 8))) == 17
        @test coord_to_index((3,4), (3, (2,2)), (2, (16, 8))) == 29
    end

    @testset "Default Stride" begin
        @test_throws DimensionMismatch coord_to_index(5, (3,4))

        @test coord_to_index((1,2), (3,4)) == 4
        @test coord_to_index((2,3), (3,4)) == 8
        @test coord_to_index((3,4), (3,4)) == 12
    end
end
