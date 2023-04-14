using CuTe, Test

@testset "zip2_by" begin
    CuTe.zip2_by((('A','a'),(('B','b'),('C','c')),'d'), (0,(0,0))) == (('A', ('B', 'C')), ('a', ('b', 'c'), 'd'))
end
