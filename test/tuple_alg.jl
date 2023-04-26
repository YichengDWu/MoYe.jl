using Shambles, Test, JET

@testset "Front" begin
    @test Shambles.front(((2, 3), 4)) === 2
    @test_opt Shambles.front(((2, 3), 4))
end

@testset "Back" begin
    @test Shambles.back((2, (3, 4))) === 4
    @test_opt Shambles.back(((2, 3), 4))
end

@testset "Tuple Cat" begin
    @test Shambles.tuple_cat((1, 2), (3, 4)) === (1, 2, 3, 4)
    @test_opt Shambles.tuple_cat((1, 2), (3, 4))
end

@testset "Insert" begin
    @test Shambles.insert((1, 2, 3), 10, 2) === (1, 10, 2, 3)
    @test_opt Shambles.insert((1, 2, 3), 10, 2)
end

@testset "Remove" begin
    @test Shambles.remove((1, 2, 3), 2) === (1, 3)
    @test_opt Shambles.remove((1, 2, 3), 2)
end

@testset "Replace" begin
    @test replace((1, 2, 3), 10, 2) === (1, 10, 3)
    @test_opt replace((1, 2, 3), 10, 2)
end

@testset "Unwrap" begin
    @test Shambles.unwrap(((1,))) === 1
    @test_opt Shambles.unwrap(((1,)))
end

@testset "Append" begin
    @test Shambles.append((1, 2), 3) === (1, 2, 3)
    @test_opt Shambles.append((1, 2), 3)

    @test Shambles.append((1, 2), 3, 4) === (1, 2, 3, 3)
    @test_opt Shambles.append((1, 2), 3, 4)
end

@testset "Prepend" begin
    @test Shambles.prepend((1, 2), 3) === (3, 1, 2)
    @test_opt Shambles.prepend((1, 2), 3)

    @test Shambles.prepend((1, 2), 3, 4) === (3, 3, 1, 2)
    @test_opt Shambles.prepend((1, 2), 3, 4)
end

@testset "Exclusive scan" begin
    Shambles.escan(*, (1, 2, 3, 4, 5), 10) === (10, 10, 20, 60, 240)
    @test_opt Shambles.escan(*, (1, 2, 3, 4, 5), 10)
end

@testset "zip2_by" begin
    @test Shambles.zip2_by((('A', 'a'), (('B', 'b'), ('C', 'c')), 'd'), (0, (0, 0))) ==
          (('A', ('B', 'C')), ('a', ('b', 'c'), 'd'))
    @test_opt Shambles.zip2_by((('A', 'a'), (('B', 'b'), ('C', 'c')), 'd'), (0, (0, 0)))
end
