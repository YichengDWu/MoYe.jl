using MoYe, Test, JET
using Static: One

@testset "Macro Layout" begin
    @test @Layout((2, (2, 2)), (4, (1, 2))) ==
          make_layout(static((2, (2, 2))), static((4, (1, 2))))
    @test @Layout(2, 4) == make_layout(_2, _4)

    @test @Layout((2, (2, 2))) == make_layout(static((2, (2, 2))))
    @test @Layout(2) == make_layout(_2)

    @test @Layout((2, (2, 2)), GenColMajor) == make_layout(static((2, (2, 2))), GenColMajor)
    @test @Layout(2, GenColMajor) == make_layout(_2, GenColMajor)

    @test @Layout((2, (2, 2)), GenRowMajor) == make_layout(static((2, (2, 2))), GenRowMajor)
    @test @Layout(2, GenRowMajor) == make_layout(_2, GenRowMajor)
end
@testset "Flatten" begin
    @test flatten(make_layout(((4, 3), 1), ((3, 1), 0))) ==
          make_layout((4, 3, 1), (3, 1, 0))

    @test_opt flatten(make_layout(((4, 3), 1), ((3, 1), 0)))
end
@testset "Coalesce" begin
    @test coalesce(@Layout((2, (1, 6)), (1, (6, 2)))) == @Layout(12, 1)
    @test_opt coalesce(@Layout((2, (1, 6)), (1, (6, 2))))
    @test_opt MoYe.bw_coalesce(MoYe.One(), (1,), (48,), 2, 1)

    function test_coalesce(layout)
        @test_opt coalesce(layout)
        coalesce_layout = coalesce(layout)
        @test depth(coalesce_layout) <= One()
        @test size(coalesce_layout) == size(layout)

        for i in One():size(layout)
            @test coalesce_layout(i) == layout(i)
        end
    end

    let layout = make_layout(_1, Int(0))
        test_coalesce(layout)
    end

    let layout = make_layout(_1, _1)
        test_coalesce(layout)
    end

    let layout = make_layout(tuple(_2, _4))
        test_coalesce(layout)
    end

    let layout = make_layout(tuple(_2, _4, _6))
        test_coalesce(layout)
    end

    let layout = make_layout(tuple(static(2), static(1), static(6)), tuple(static(1), static(6), static(2)))
        test_coalesce(layout)
    end

    let layout = make_layout(tuple(static(2), static(1), static(6)), tuple(static(1), 7, static(2)))
        test_coalesce(layout)
    end

    let layout = make_layout(tuple(static(2), static(1), static(6)), tuple(static(4), 7, static(8)))
        test_coalesce(layout)
    end

    let layout = make_layout(tuple(2, static(4), static(6)))
        test_coalesce(layout)
    end

    let layout = make_layout(tuple(static(2), 4, static(6)))
        test_coalesce(layout)
    end

    let layout = make_layout(tuple(static(2), static(4), 6))
        test_coalesce(layout)
    end

    let layout = make_layout(tuple(static(2), static(4)), GenRowMajor)
        test_coalesce(layout)
    end

    let layout = make_layout(tuple(static(2), static(4), static(6)), GenRowMajor)
        test_coalesce(layout)
    end

    let layout = make_layout(tuple(2, static(4), static(6)), GenRowMajor)
        test_coalesce(layout)
    end

    let layout = make_layout(tuple(static(2), 4, static(6)), GenRowMajor)
        test_coalesce(layout)
    end

    let layout = make_layout(tuple(static(2), static(4), 6), GenRowMajor)
        test_coalesce(layout)
    end

    let layout = make_layout(tuple(static(2), static(1), static(3)), GenRowMajor)
        test_coalesce(layout)
    end

    let layout = make_layout(tuple(static(2), 1, static(3)), GenRowMajor)
        test_coalesce(layout)
    end

    let layout = make_layout(tuple(static(2), 1, static(3)), tuple(static(2), 4, static(4)))
        test_coalesce(layout)
    end

    let layout = make_layout(tuple(static(2), 1, static(3)), tuple(static(2), Int(0), static(4)))
        test_coalesce(layout)
    end

    let layout = Layout(tuple(tuple(static(2), static(2)), tuple(static(2), static(2))),
                       tuple(tuple(static(1), static(4)), tuple(static(8), static(32))))
        test_coalesce(layout)
    end

end

@testset "Composition" begin
    @test make_layout(20, 2) ∘ make_layout((4, 5), (1, 4)) == make_layout((4, 5), (2, 8))
    @test make_layout(20, 2) ∘ make_layout((4, 5), (5, 1)) == make_layout((4, 5), (10, 2))

    @test_opt make_layout(20, 2) ∘ make_layout((4, 5), (1, 4))
    @test_opt make_layout(20, 2) ∘ make_layout((4, 5), (5, 1))

    function test_composition(A,B)
        @test_opt A ∘ B
        C = A ∘ B
        @test MoYe.iscompatible(B,C)
        for i in static(1):size(C)
            @test C(i) == A(B(i))
        end
    end

    let a = @Layout(1,0), b = @Layout(1,0)
        test_composition(a, b)
    end

    let a = @Layout(1,0), b = @Layout(1,1)
        test_composition(a, b)
    end

    let a = @Layout(1,1), b = @Layout(1,0)
        test_composition(a, b)
    end

    let a = @Layout(1,1), b = @Layout(1,1)
        test_composition(a, b)
    end

    let a = @Layout((4,)), b = @Layout((4,))
        test_composition(a, b)
    end

    let a = @Layout((4,), (2,)), b = @Layout((4,))
        test_composition(a, b)
    end

    let a = @Layout((4,), (0,)), b = @Layout((4,))
        test_composition(a, b)
    end

    let a = @Layout((4,)), b = @Layout((4,), (0,))
        test_composition(a, b)
    end

    let a = @Layout((4,)), b = @Layout((1,), (0,))
        test_composition(a, b)
    end

    let a = @Layout((4,)), b = @Layout((2,))
        test_composition(a, b)
    end

    let a = @Layout((4,), (2,)), b = @Layout((2,))
        test_composition(a, b)
    end

    let a = @Layout((4,)), b = @Layout((2,), (2,))
        test_composition(a, b)
    end

    let a = @Layout((4,), (2,)), b = @Layout((2,), (2,))
        test_composition(a, b)
    end

    let a = @Layout((4,3)), b = @Layout((12,))
        test_composition(a, b)
    end

    let a = @Layout((12,)), b = @Layout((4,3))
        test_composition(a, b)
    end

    let a = @Layout((12,), (2,)), b = @Layout((4,3))
        test_composition(a, b)
    end

    let a = @Layout((12,)), b = @Layout((4,3), (3,1))
        test_composition(a, b)
    end

    let a = @Layout((12,), (2,)), b = @Layout((4,3), (3,1))
        test_composition(a, b)
    end

    let a = @Layout((12,)), b = @Layout((2,3), (2,4))
        test_composition(a, b)
    end

    let a = @Layout((4,3)), b = @Layout((4,3))
        test_composition(a, b)
    end

    let a = @Layout((4,3)), b = @Layout((6,), (2,))
        test_composition(a, b)
    end

    let a = @Layout((4,3)), b = @Layout((6,2), (2,1))
        test_composition(a, b)
    end

    let a = @Layout((4,3), (3,1)), b = @Layout((4,3))
        test_composition(a, b)
    end

    let a = @Layout((4,3), (3,1)), b = @Layout((12,))
        test_composition(a, b)
    end

    let a = @Layout((4,3), (3,1)), b = @Layout((6,), (2,))
        test_composition(a, b)
    end

    let a = @Layout((4,3), (3,1)), b = @Layout((6,2), (2,1))
        test_composition(a, b)
    end

    let a = @Layout((8,8)), b = @Layout(((2,2,2),(2,2,2)), ((1,16,4),(8,2,32)))
        test_composition(a, b)
    end

    let a = @Layout((8,8), (8,1)), b = @Layout(((2,2,2),(2,2,2)), ((1,16,4),(8,2,32)))
        test_composition(a, b)
    end

    let a = @Layout(((4,2),), ((1,16),)), b = @Layout((4,2), (2,1))
        test_composition(a, b)
    end

    let a = @Layout((2,2), (2,1)), b = @Layout((2,2), (2,1))
        test_composition(a, b)
    end

    let a = @Layout((4,8,2)), b = @Layout((2,2,2), (2,8,1))
        test_composition(a, b)
    end

    let a = @Layout((4,8,2), (2,8,1)), b = @Layout((2,2,2), (1,8,2))
        test_composition(a, b)
    end

    let a = @Layout((4,8,2), (2,8,1)), b = @Layout((4,2,2), (2,8,1))
        test_composition(a, b)
    end

    @testset "Dynamic" begin
        let a = make_layout(12, 1), b = make_layout(static(4), static(1))
            test_composition(a, b)
        end

        let a = make_layout(12, 1), b = make_layout(static(4), 1)
            test_composition(a, b)
        end

        let a = make_layout(12, static(1)), b = make_layout(static(4), 1)
            test_composition(a, b)
        end

        let a = make_layout(12, static(1)), b = make_layout(static(4), static(1))
            test_composition(a, b)
        end

        let a = make_layout(tuple(12, 3), tuple(1, 24)), b = make_layout(tuple(static(4)), tuple(static(1)))
            test_composition(a, b)
        end

        let a = make_layout(16, 2), b = make_layout(4, 2)
            test_composition(a, b)
        end

        let a = make_layout(tuple(128, 24, 5), tuple(1, 128, 3072)), b = make_layout(64, 2)
            test_composition(a, b)
        end

        let a = make_layout(tuple(128, 24, 5), tuple(1, 128, 3072)), b = make_layout(480, static(32))
            test_composition(a, b)
        end
    end
end

@testset "Complement" begin
    @test complement(@Layout(4, 1), static(24)) == @Layout(6, 4)
    @test complement(@Layout(6, 4), static(24)) == @Layout(4, 1)

    @test_opt complement(@Layout(4, 1), static(24))
    @test_opt complement(@Layout(6, 4), static(24))

    function test_complement(l, cotarget)
        @test_opt  complement(l, cotarget)
        result = complement(l, cotarget)
        @test size(result) ≥ cotarget ÷ size(filter(l))
        @test cosize(result) ≤ cld(cotarget, cosize(l)) * cosize(l)

        completed = make_layout(l, result)
        @test cosize(completed) ≥ cotarget

        for i in 2:size(result)
            @test result(i-1) < result(i)
            for j in 1:size(l)
                @test result(i) != l(j)
            end
        end

        @test size(result) ≤ cosize(result)
        @test cosize(result) ≥ cotarget ÷ size(filter(l))
        @test cosize(completed) ≤ cosize(result) + cosize(l)
        @test cosize(result) ≥  cotarget ÷ size(filter(l))

        if MoYe.dynamic(MoYe.is_static(stride(make_layout(l, result))))
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
        test_complement(layout, One())
        test_complement(layout, static(2))
        test_complement(layout, static(8))
    end

    let layout = @Layout(4,0)
        test_complement(layout, One())
        test_complement(layout, static(2))
        test_complement(layout, static(8))
    end

    let layout = @Layout(4,1)
        test_complement(layout, One())
        test_complement(layout, static(2))
        test_complement(layout, static(8))
    end

    let layout = @Layout(4,2)
        test_complement(layout, One())
        test_complement(layout)
        test_complement(layout, static(16))
    end

    let layout = @Layout(4,4)
        test_complement(layout, One())
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
    tile = @Layout((2, 2), (1, 2))
    matrix_of_tiles = @Layout((3, 4), (4, 1))

    @testset "Logical product" begin
        result = logical_product(tile, matrix_of_tiles)
        @test shape(result) == ((2, 2), (3, 4))
        @test stride(result) == ((1, 2), (16, 4))

        @test_opt logical_product(tile, matrix_of_tiles)

        @test logical_product(@Layout(1), @Layout((2,2))) == @Layout((1,(2,2)))

        function test_logical_product(A,B)
            @test_opt logical_product(A,B)
            C = logical_product(A,B)
            @test rank(C) == 2
            @test MoYe.iscompatible(A, first(C))
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

    @testset "Zipped product" begin
        result = zipped_product(tile, matrix_of_tiles)
        @test shape(result) == ((2, 2), (3, 4))
        @test stride(result) == ((1, 2), (16, 4))

        @test_opt zipped_product(static(tile), static(matrix_of_tiles))
    end

    @testset "Tiled product" begin
        result = tiled_product(tile, matrix_of_tiles)
        @test shape(result) == ((2, 2), 3, 4)
        @test stride(result) == ((1, 2), 16, 4)

        @test_opt tiled_product(static(tile), static(matrix_of_tiles))
    end
end

@testset "Division" begin
    tile = @Layout((2, 2), (1, 2))
    matrix_of_tiles = @Layout((3, 4), (4, 1))
    raked_prod = raked_product(tile, matrix_of_tiles)
    subtile = (@Layout(2, 3), @Layout(2, 4))

    @testset "Logical division" begin
        @test logical_divide(@Layout(16, 3), @Layout(4, 1)) == @Layout((4, 4), (3, 12))
        @test logical_divide(@Layout(16, 3), @Layout(4, 4)) == @Layout((4, 4), (12, 3))
        @test logical_divide(@Layout(16, 3), @Layout(4, 2)) ==
              @Layout((4, (2, 2)), (6, (3, 24)))
        @test logical_divide(@Layout(16, 3), @Layout((2, 2), (4, 1))) ==
              @Layout(tuple((2, 2), (2, 2)), tuple((12, 3), (6, 24)))
        @test logical_divide(raked_prod, subtile) ==
              @Layout(((2, 3), (2, 4)), ((1, 16), (2, 4)))

        @test_opt logical_divide(raked_prod, subtile)
        @test_call logical_divide(raked_prod, subtile)

        function test_logical_divide(A, B)
            @test_opt logical_divide(A,B)
            C = logical_divide(A,B)
            @test rank(C) == 2
            @test MoYe.iscompatible(B, first(C))
        end

        let layout = @Layout(1, 0),
            tile   = @Layout(1, 0)
            test_logical_divide(layout, tile)
        end

        let layout = @Layout(1, 0),
            tile   = @Layout(1, 1)
            test_logical_divide(layout, tile)
        end

        let layout = @Layout(1, 1),
            tile   = @Layout(1, 0)
            test_logical_divide(layout, tile)
        end

        let layout = @Layout(1, 1),
            tile   = @Layout(1, 1)
            test_logical_divide(layout, tile)
        end

        let layout = @Layout(6, 1),
            tile   = @Layout(2, 1)
            test_logical_divide(layout, tile)
        end

        let layout = @Layout(6, 1),
            tile   = @Layout(2, 3)
            test_logical_divide(layout, tile)
        end

        let layout = @Layout((6, 6), (1, 12)),
            tile   = @Layout((6, 3), (3, 1))
            test_logical_divide(layout, tile)
        end

        let layout = @Layout((6, 6), (12, 1)),
            tile   = @Layout((6, 3), (3, 1))
            test_logical_divide(layout, tile)
        end

        let layout = @Layout(32),
            tile   = @Layout((2, 8))
            test_logical_divide(layout, tile)
        end

        let layout = @Layout((4, 1), (1, 1)),
            tile   = @Layout(2, 1)
            test_logical_divide(layout, tile)
        end

        let layout = @Layout((4, 1), (1, 1)),
            tile   = @Layout(2, 2)
            test_logical_divide(layout, tile)
        end

        let layout = @Layout((8, 8), (1, 8)),
            tile   = @Layout((32, 2))
            test_logical_divide(layout, tile)
        end

        let layout = @Layout((8, 8), (8, 1)),
            tile   = @Layout((32, 2))
            test_logical_divide(layout, tile)
        end
    end

    @testset "Zipped division" begin
        @test zipped_divide(raked_prod, subtile) ==
              @Layout(((2, 2), (3, 4)), ((1, 2), (16, 4)))
        @test_opt zipped_divide(static(raked_prod), static(subtile))
    end

    @testset "Tiled division" begin
        @test tiled_divide(raked_prod, subtile) ==
              @Layout(((2, 2), 3, 4), ((1, 2), 16, 4))
        @test_opt zipped_divide(static(raked_prod), static(subtile))
    end
end

@testset "Inverse" begin
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
        test_right_inverse(@Layout((2,4,4,6), (4,1,0,8)))
        test_right_inverse(@Layout((4,2), (1,16)))
        test_right_inverse(@Layout((4,2), (1,5)))
    end
end
