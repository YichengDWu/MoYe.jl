using CuTe, Test, JET

CuTe.static(l::Layout) = Layout(static(shape(l)), static(stride(l)))

@testset "Macro" begin
    @test @Layout((2, (2, 2)), (4, (1, 2))) ==
          make_layout(static((2, (2, 2))), static((4, (1, 2))))
    @test @Layout(2, 4) == make_layout(static(2), static(4))

    @test @Layout((2, (2, 2))) == make_layout(static((2, (2, 2))))
    @test @Layout(2) == make_layout(static(2))

    @test @Layout((2, (2, 2)), GenColMajor) == make_layout(static((2, (2, 2))), GenColMajor)
    @test @Layout(2, GenColMajor) == make_layout(static(2), GenColMajor)

    @test @Layout((2, (2, 2)), GenRowMajor) == make_layout(static((2, (2, 2))), GenRowMajor)
    @test @Layout(2, GenRowMajor) == make_layout(static(2), GenRowMajor)
end
@testset "Flatten" begin
    @test flatten(make_layout(((4, 3), 1), ((3, 1), 0))) ==
          make_layout((4, 3, 1), (3, 1, 0))

    @test_opt flatten(make_layout(((4, 3), 1), ((3, 1), 0)))
end
@testset "Coalesce" begin
    @test coalesce(make_layout((2, (1, 6)), (1, (6, 2)))) == make_layout(12, 1)
    @test_opt coalesce(make_layout((2, (1, 6)), (1, (6, 2))))
end

@testset "Composition" begin
    @test make_layout(20, 2) ∘ make_layout((4, 5), (1, 4)) == make_layout((4, 5), (2, 8))
    @test make_layout(20, 2) ∘ make_layout((4, 5), (5, 1)) == make_layout((4, 5), (10, 2))

    @test_opt make_layout(20, 2) ∘ make_layout((4, 5), (1, 4))
    @test_opt make_layout(20, 2) ∘ make_layout((4, 5), (5, 1))
end

@testset "Complement" begin
    @test complement(make_layout(4, 1), 24) == make_layout(6, 4)
    @test complement(make_layout(6, 4), 24) == make_layout(4, 1)

    @test_opt complement(make_layout(4, 1), 24)
    @test_opt complement(make_layout(6, 4), 24)
end

@testset "Product" begin
    tile = make_layout((2, 2), (1, 2))
    matrix_of_tiles = make_layout((3, 4), (4, 1))

    @testset "Logical product" begin
        result = logical_product(tile, matrix_of_tiles)
        @test shape(result) == ((2, 2), (3, 4))
        @test stride(result) == ((1, 2), (16, 4))

        @test_opt logical_product(static(tile), static(matrix_of_tiles)) # note that `complement` requires a static layout to avoid dynamic dispatch
    end
    @testset "Blocked product" begin
        result = blocked_product(tile, matrix_of_tiles, true)
        @test shape(result) == ((2, 3), 8)
        @test stride(result) == ((1, 16), 2)

        @test_opt blocked_product(static(tile), static(matrix_of_tiles), true)
    end
    @testset "Raked product" begin
        result = raked_product(tile, matrix_of_tiles, true)
        @test shape(result) == ((3, 2), (4, 2))
        @test stride(result) == ((16, 1), (4, 2))

        @test_opt raked_product(static(tile), static(matrix_of_tiles), true)
    end
end

@testset "Division" begin
    tile = make_layout((2, 2), (1, 2))
    matrix_of_tiles = make_layout((3, 4), (4, 1))
    raked_prod = raked_product(tile, matrix_of_tiles)
    subtile = (Layout(2, 3), Layout(2, 4))

    @testset "Logical division" begin
        @test logical_divide(Layout(16, 3), Layout(4, 1)) == Layout((4, 4), (3, 12))
        @test logical_divide(Layout(16, 3), Layout(4, 4)) == Layout((4, 4), (12, 3))
        @test logical_divide(Layout(16, 3), Layout(4, 2)) ==
              Layout((4, (2, 2)), (6, (3, 24)))
        @test logical_divide(Layout(16, 3), Layout((2, 2), (4, 1))) ==
              Layout(tuple((2, 2), (2, 2)), tuple((12, 3), (6, 24)))
        @test logical_divide(raked_prod, subtile) ==
              make_layout(((2, 3), (2, 4)), ((1, 16), (2, 4)))

        @test_opt logical_divide(static(raked_prod), static(subtile))
        @test_call logical_divide(raked_prod, subtile)
    end

    @testset "Zipped division" begin
        @test zipped_divide(raked_prod, subtile) ==
              make_layout(((2, 2), (3, 4)), ((1, 2), (16, 4)))
        @test_opt zipped_divide(static(raked_prod), static(subtile))
    end
end
