using Test, SafeTestsets

@safetestset "Tuple Algorithms" begin include("tuple_alg.jl") end
@safetestset "IntTuple" begin include("int_tuple.jl") end
@safetestset "Stride" begin include("stride.jl") end
@safetestset "Layout" begin include("layout.jl") end
@safetestset "Static" begin include("static.jl") end
@safe @testset "Engine" begin include("engine.jl") end
