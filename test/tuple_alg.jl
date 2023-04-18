using CuTe, Test, JET

@testset "Front" begin
    @test CuTe.front(((2,3),4)) === 2
    @test_opt CuTe.front(((2,3),4))
end

@testset "Back" begin
    @test CuTe.back((2,(3,4))) === 4
    @test_opt CuTe.back(((2,3),4))
end

@testset "Tuple Cat" begin
    @test CuTe.tuple_cat((1,2), (3,4)) === (1,2,3,4)
    @test_opt CuTe.tuple_cat((1,2), (3,4))
end

@testset "Insert" begin
    @test CuTe.insert((1,2,3), 10, 2) === (1,10,2,3)
    @test_opt CuTe.insert((1,2,3), 10, 2)
end

@testset "Remove" begin
    @test CuTe.remove((1,2,3), 2) === (1,3)
    @test_opt CuTe.remove((1,2,3), 2)
end

@testset "Replace" begin
    @test replace((1,2,3), 10, 2) === (1,10,3)
    @test_opt replace((1,2,3), 10, 2)
end


@testset "Unwrap" begin
    @test CuTe.unwrap(((1,))) === 1
    @test_opt CuTe.unwrap(((1,)))
end

@testset "Append" begin
    @test CuTe.append((1,2), 3) === (1,2,3)
    @test_opt CuTe.append((1,2), 3)

    @test CuTe.append((1,2), 3, 4) === (1,2,3,3)
    @test_opt CuTe.append((1,2), 3, 4)
end

@testset "Prepend" begin
    @test CuTe.prepend((1,2), 3) === (3,1,2)
    @test_opt CuTe.prepend((1,2), 3)

    @test CuTe.prepend((1,2), 3, 4) === (3,3,1,2)
    @test_opt CuTe.prepend((1,2), 3, 4)
end

@testset "Exclusive scan" begin
    CuTe.escan(*, (1,2,3,4,5), 10) === (10, 10, 20, 60, 240)
    @test_opt CuTe.escan(*, (1,2,3,4,5), 10)
end

@testset "zip2_by" begin
    @test CuTe.zip2_by((('A', 'a'), (('B', 'b'), ('C', 'c')), 'd'), (0, (0, 0))) ==
          (('A', ('B', 'C')), ('a', ('b', 'c'), 'd'))
    @test_opt CuTe.zip2_by((('A', 'a'), (('B', 'b'), ('C', 'c')), 'd'), (0, (0, 0)))
end
