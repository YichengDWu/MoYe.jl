using CuTe, Test, JET

@testset "Coordinate to Index" begin
    @testset "1-D Coordinate" begin
        @test coord_to_index(5, (3, 4), (2, 8)) == 11
        @test coord_to_index(7, (3, 4), (2, 8)) == 17
        @test coord_to_index(12, (3, 4), (2, 8)) == 29

        @test_opt coord_to_index(5, (3, 4), (2, 8))
        @test_opt coord_to_index(7, (3, 4), (2, 8))
        @test_opt coord_to_index(12, (3, 4), (2, 8))
    end

    @testset "h-D Coordinate" begin
        @test coord_to_index((2, 2), (3, 4), (2, 8)) == 11
        @test coord_to_index((1, 3), (3, 4), (2, 8)) == 17
        @test coord_to_index((3, 4), (3, 4), (2, 8)) == 29

        @test_opt coord_to_index((2, 2), (3, 4), (2, 8))
        @test_opt coord_to_index((1, 3), (3, 4), (2, 8))
        @test_opt coord_to_index((3, 4), (3, 4), (2, 8))
    end

    @testset "R-D Coordinate" begin
        @test coord_to_index((2, 3), (3, (2, 2)), (2, (16, 8))) == 11
        @test coord_to_index((1, 2), (3, (2, 2)), (2, (16, 8))) == 17
        @test coord_to_index((3, 4), (3, (2, 2)), (2, (16, 8))) == 29

        @test_opt coord_to_index((2, 3), (3, (2, 2)), (2, (16, 8)))
        @test_opt coord_to_index((1, 2), (3, (2, 2)), (2, (16, 8)))
        @test_opt coord_to_index((3, 4), (3, (2, 2)), (2, (16, 8)))
    end

    @testset "Default Stride" begin
        @test_throws DimensionMismatch coord_to_index(5, (3, 4))

        @test coord_to_index((1, 2), (3, 4)) == 4
        @test coord_to_index((2, 3), (3, 4)) == 8
        @test coord_to_index((3, 4), (3, 4)) == 12

        @test_opt coord_to_index((1, 2), (3, 4))
        @test_opt coord_to_index((2, 3), (3, 4))
        @test_opt coord_to_index((3, 4), (3, 4))
    end
end

@testset "Index to Coord" begin
    @test index_to_coord(9, (3, 4), (1, 3)) == (3, 3)
    @test index_to_coord(10, (3, 4), (1, 3)) == (1, 4)

    @test_opt index_to_coord(9, (3, 4), (1, 3))
    @test_opt index_to_coord(10, (3, 4), (1, 3))

    @test_throws DimensionMismatch index_to_coord(9, (3, 4), (1, 3, 5))
end
