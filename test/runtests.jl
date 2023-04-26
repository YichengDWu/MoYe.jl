using Test, SafeTestsets

@safetestset "Tuple Algorithms" begin include("tuple_alg.jl") end
@safetestset "IntTuple" begin include("int_tuple.jl") end
@safetestset "Stride" begin include("stride.jl") end
@safetestset "Layout" begin include("layout.jl") end
@safetestset "Static" begin include("static.jl") end
@safetestset "Engine" begin include("engine.jl") end
@testset "CuTeArray" begin
    @safetestset "CuTeArray" begin include("cutearray.jl") end
    @safetestset "Broadcast" begin include("broadcast.jl") end
end

@testset "Device" begin
    @safetestset "Memory" begin include("device/memory.jl") end
    @safetestset "MMA" begin include("device/mmaop.jl") end
    @safetestset "MMATraits" begin include("device/mmatraits.jl") end
    @safetestset "Pointer" begin include("device/pointer.jl") end
end
