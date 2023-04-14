using CuTe, Test

@testset "IntTuple" begin include("int_tuple.jl") end

@testset "Stride" begin include("stride.jl") end

@testset "Layout" begin include("layout.jl") end
