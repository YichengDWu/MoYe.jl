using Moye, Test, JET

@testset "Front" begin
    @test Moye.front(((2, 3), 4)) === 2
    @test_opt Moye.front(((2, 3), 4))
end

@testset "Back" begin
    @test Moye.back((2, (3, 4))) === 4
    @test_opt Moye.back(((2, 3), 4))
end

@testset "Tuple Cat" begin
    @test Moye.tuple_cat((1, 2), (3, 4)) === (1, 2, 3, 4)
    @test_opt Moye.tuple_cat((1, 2), (3, 4))
end

@testset "Insert" begin
    @test Moye.insert((1, 2, 3), 10, 2) === (1, 10, 2, 3)
    @test_opt Moye.insert((1, 2, 3), 10, 2)
end

@testset "Remove" begin
    @test Moye.remove((1, 2, 3), 2) === (1, 3)
    @test_opt Moye.remove((1, 2, 3), 2)
end

@testset "Replace" begin
    @test replace((1, 2, 3), 10, 2) === (1, 10, 3)
    @test_opt replace((1, 2, 3), 10, 2)
end

@testset "Unwrap" begin
    @test Moye.unwrap(((1,))) === 1
    @test_opt Moye.unwrap(((1,)))
end

@testset "Append" begin
    @test Moye.append((1, 2), 3) === (1, 2, 3)
    @test_opt Moye.append((1, 2), 3)

    @test Moye.append((1, 2), 3, 4) === (1, 2, 3, 3)
    @test_opt Moye.append((1, 2), 3, 4)
end

@testset "Prepend" begin
    @test Moye.prepend((1, 2), 3) === (3, 1, 2)
    @test_opt Moye.prepend((1, 2), 3)

    @test Moye.prepend((1, 2), 3, 4) === (3, 3, 1, 2)
    @test_opt Moye.prepend((1, 2), 3, 4)
end

@testset "Exclusive scan" begin
    Moye.escan(*, (1, 2, 3, 4, 5), 10) === (10, 10, 20, 60, 240)
    @test_opt Moye.escan(*, (1, 2, 3, 4, 5), 10)
end

@testset "zip2_by" begin
    @test Moye.zip2_by((('A', 'a'), (('B', 'b'), ('C', 'c')), 'd'), (0, (0, 0))) ==
          (('A', ('B', 'C')), ('a', ('b', 'c'), 'd'))
    @test_opt Moye.zip2_by((('A', 'a'), (('B', 'b'), ('C', 'c')), 'd'), (0, (0, 0)))
end

@testset "Make Tuple" begin
    t = typeof(static((1,2,3)))
    t2 = make_tuple(t)
    @test t2 === static((1,2,3))

    t3 = typeof(static((1,2,(3,4))))
    t4 = make_tuple(t3)
    @test t4 === static((1,2,(3,4)))

    t5 = typeof(static((1,2,(3,4),5)))
    t6 = make_tuple(t5)
    @test t6 === static((1,2,(3,4),5))
end
