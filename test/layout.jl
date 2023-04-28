using Shambles, Test, JET

Shambles.static(l::Layout) = Layout(static(shape(l)), static(stride(l)))

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

    function test_complement(l, cosize_hi)
        result = complement(l, cosize_hi)
        @test size(result) ≥ cosize_hi ÷ size(filter(l))
        @test cosize(result) ≤ cld(cosize_hi, cosize(l)) * cosize(l)

        for i in 2:size(result)
            @test result(i-1) < result(i)
            for j in 1:size(l)
                @test result(i) != l(j)
            end
        end

        @test size(result) ≤ cosize(result)
        @test cosize(result) ≥ cosize_hi ÷ size(filter(l))

        if Shambles.Static.dynamic(Shambles.Static.is_static(stride(make_layout(l, result))))
            @test size(complement(make_layout(l, result))) == 1
        end
    end

    test_complement(l::Layout) = test_complement(l, cosize(l))

    let layout = @Layout(1,0)
        test_complement(layout)
        test_complement(layout, static(2))
    end

    let layout = @Layout(1,1)
        test_complement(layout)
        test_complement(layout, static(2))
    end

    let layout = @Layout(1,2)
        test_complement(layout, static(1))
        test_complement(layout, static(2))
        test_complement(layout, static(8))
    end

    let layout = @Layout(4,0)
        test_complement(layout, static(1))
        test_complement(layout, static(2))
        test_complement(layout, static(8))
    end

    let layout = @Layout(4,1)
        test_complement(layout, static(1))
        test_complement(layout, static(2))
        test_complement(layout, static(8))
    end

    let layout = @Layout(4,2)
        test_complement(layout, static(1))
        test_complement(layout)
        test_complement(layout, static(16))
    end

    let layout = @Layout(4,4)
        test_complement(layout, static(1))
        test_complement(layout)
        test_complement(layout, static(17))
    end

    let layout = @Layout(2,4)
        test_complement(layout)
    end

    let layout = @Layout(2,3)
        test_complement(layout)
    end

    let layout = @Layout((2,4), (1,4))
        test_complement(layout)
    end

    let layout = @Layout((2,4,8), (8,1,64))
        test_complement(layout)
    end

    let layout = @Layout((2,4,8), (8,1,0))
        test_complement(layout)
        test_complement(layout, static(460))
    end

    let layout = @Layout(((2,2), (2,2)), ((1,4), (8,32)))
        test_complement(layout)
    end

end

@testset "Product" begin
    tile = make_layout((2, 2), (1, 2))
    matrix_of_tiles = make_layout((3, 4), (4, 1))

    @testset "Logical product" begin
        result = logical_product(tile, matrix_of_tiles)
        @test shape(result) == ((2, 2), (3, 4))
        @test stride(result) == ((1, 2), (16, 4))

        @test_opt logical_product(static(tile), static(matrix_of_tiles)) # note that `complement` requires a static layout to avoid dynamic dispatch

        function test_logical_product(A,B)
            C = logical_product(A,B)
            @test rank(C) == 2
            @test Shambles.iscompatible(A, first(C))
        end

        let vec = @Layout(1,0), tile = @Layout(1,0)
            test_logical_product(vec, tile)
        end

        let vec = @Layout(1,1), tile = @Layout(1,0)
            test_logical_product(tile, vec)
        end

        let vec = @Layout(1,1), tile = @Layout(1,1)
            test_logical_product(vec, tile)
        end

        let vec = @Layout(3,1), tile = @Layout(4,0)
            test_logical_product(tile, vec)
        end

        let vec = @Layout(3,0), tile = @Layout(4,0)
            test_logical_product(vec, tile)
        end

        let vec = @Layout(3,2), tile = @Layout(4,1)
            test_logical_product(vec, tile)
        end

        let vec = @Layout((3,)), tile = @Layout((2,4))
            test_logical_product(vec, tile)
        end

        let vec = @Layout((8,(2,2))), tile = @Layout(4,2)
            test_logical_product(vec, tile)
        end

        let vec = @Layout((2,2)), tile = @Layout((3,3), (3,1))
            test_logical_product(vec, tile)
        end

        let vec = @Layout(3,32), tile = @Layout((8,8))
            test_logical_product(vec, tile)
        end

        let vec = @Layout(((2,2),(2,2)), ((1,4),(8,32))), tile = @Layout((2,2), (1,2))
            test_logical_product(vec, tile)
        end

        let vec = @Layout(((2,2),(2,2)), ((1,4),(8,32))), tile = @Layout((2,2), (2,1))
            test_logical_product(vec, tile)
        end
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

        function test_logical_divide(A,B)
            C = logical_divide(A,B)
            @test rank(C) == 2
            @test Shambles.iscompatible(B, first(C))
        end

        let vec = @Layout(1,0), tile = @Layout(1,0)
            test_logical_divide(vec, tile)
        end

        let vec = @Layout(1,0), tile = @Layout(1,1)
            test_logical_divide(tile, vec)
        end

        let vec = @Layout(1,1), tile = @Layout(1,0)
            test_logical_divide(vec, tile)
        end

        let vec = @Layout(6,1), tile = @Layout(2,1)
            test_logical_divide(vec, tile)
        end

        let vec = @Layout(6,1), tile = @Layout(2,3)
            test_logical_divide(vec, tile)
        end

        let vec = @Layout(6,1), tile = @Layout((2,3),(3,1))
            test_logical_divide(vec, tile)
        end

        let vec = @Layout(6,2), tile = @Layout(2,1)
            test_logical_divide(vec, tile)
        end

        let vec = @Layout(6,2), tile = @Layout(2,3)
            test_logical_divide(vec, tile)
        end

        let vec = @Layout(6,2), tile = @Layout((2,3),(3,1))
            test_logical_divide(vec, tile)
        end

        let vec = @Layout((6,6),(1,12)), tile = @Layout((6,3),(3,1))
            test_logical_divide(vec, tile)
        end

        let vec = @Layout((6,6),(12,1)), tile = @Layout((6,3),(3,1))
            test_logical_divide(vec, tile)
        end
    end

    @testset "Zipped division" begin
        @test zipped_divide(raked_prod, subtile) ==
              make_layout(((2, 2), (3, 4)), ((1, 2), (16, 4)))
        @test_opt zipped_divide(static(raked_prod), static(subtile))
    end
end

@tesetset "Inverse" begin
    @testset "Right Inverse" begin
        function test_right_inverse(l)
            inv_l = right_inverse(l)

            @test_opt right_inverse(l)
            @test_call right_inverse(l)

            for i in 1:size(inv_l)
                @test l(inv_l(i)) == i
            end
        end

        test_right_inverse(@Layout(1, 0))
        test_right_inverse(@Layout(1, 1))
        test_right_inverse(@Layout((4,), (0,)))
        test_right_inverse(@Layout((4,), (1,)))
        test_right_inverse(@Layout((4,), (2,)))

        test_right_inverse(@Layout((1,1), (0,0)))
        test_right_inverse(@Layout((3,7), (0,0)))
        test_right_inverse(@Layout((1,), (1,)))
        test_right_inverse(@Layout((2,4), (0,2)))
        test_right_inverse(@Layout((8,4)))
        test_right_inverse(@Layout((8,4), (4,1)))
        test_right_inverse(@Layout((2,4,6)))
        test_right_inverse(@Layout((2,4,6), (4,1,8)))
        # test_right_inverse(@Layout((2,4,4,6), (4,1,0,8))) failed to optimize due to recursion
        test_right_inverse(@Layout((4,2), (1,16)))
        test_right_inverse(@Layout((4,2), (1,5)))
    end
end
