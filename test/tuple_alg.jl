using MoYe, Test, JET
using MoYe: make_tuple
@testset "Unflatten" begin
    @test unflatten((1, 2, 3), (0, 0, 0)) == (1, 2, 3)
    @test unflatten((1, 2, 3, 4), (0, (0, (0, 0)))) == (1, (2, (3, 4)))
    @test unflatten((1, 2, 3, 4), ((0, 0), (0, 0))) == ((1, 2), (3, 4))
    @test unflatten((1, 2, 3, 4, 5), (0, (0, 0), 0)) == (1, (2, 3), 4)
    @test unflatten((1, 2, 3, 4, 5, 6), (0, (0, (0, (0, (0, 0)))))) == (1, (2, (3, (4, (5, 6)))))
    @test unflatten((), ()) == ()
end

@testset "Front" begin
    @test MoYe.front(((2, 3), 4)) === 2
    @test_opt MoYe.front(((2, 3), 4))
end

@testset "Back" begin
    @test MoYe.back((2, (3, 4))) === 4
    @test_opt MoYe.back(((2, 3), 4))
end

@testset "Tuple Cat" begin
    @test MoYe.tuple_cat((1, 2), (3, 4)) === (1, 2, 3, 4)
    @test_opt MoYe.tuple_cat((1, 2), (3, 4))
end

@testset "Insert" begin
    @test MoYe.insert((1, 2, 3), 10, 2) === (1, 10, 2, 3)
    @test_opt MoYe.insert((1, 2, 3), 10, 2)
end

@testset "Remove" begin
    @test MoYe.remove((1, 2, 3), static(2)) === (1, 3)
    @test_opt MoYe.remove((1, 2, 3), static(2))
end

@testset "Replace" begin
    @test replace((1, 2, 3), 10, 2) === (1, 10, 3)
    @test_opt replace((1, 2, 3), 10, 2)
end

@testset "Unwrap" begin
    @test MoYe.unwrap(((1,))) === 1
    @test_opt MoYe.unwrap(((1,)))
end

@testset "Append" begin
    @test MoYe.append((1, 2), 3) === (1, 2, 3)
    @test_opt MoYe.append((1, 2), 3)

    @test MoYe.append((1, 2), 3, static(4)) === (1, 2, 3, 3)
    @test_opt MoYe.append((1, 2), 3, static(4))
end

@testset "Prepend" begin
    @test MoYe.prepend((1, 2), 3) === (3, 1, 2)
    @test_opt MoYe.prepend((1, 2), 3)

    @test MoYe.prepend((1, 2), 3, static(4)) === (3, 3, 1, 2)
    @test_opt MoYe.prepend((1, 2), 3, static(4))
end

@testset "Exclusive scan" begin
    MoYe.escan(*, (1, 2, 3, 4, 5), 10) === (10, 10, 20, 60, 240)
    @test_opt MoYe.escan(*, (1, 2, 3, 4, 5), 10)
end

@testset "zip2_by" begin
    @test MoYe.zip2_by((('A', 'a'), (('B', 'b'), ('C', 'c')), 'd'), (0, (0, 0))) ==
          (('A', ('B', 'C')), ('a', ('b', 'c'), 'd'))
    @test_opt MoYe.zip2_by((('A', 'a'), (('B', 'b'), ('C', 'c')), 'd'), (0, (0, 0)))
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
