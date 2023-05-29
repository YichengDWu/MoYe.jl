using Test, SafeTestsets

@safetestset "Tuple Algorithms" begin include("tuple_alg.jl") end
@safetestset "IntTuple" begin include("int_tuple.jl") end
@safetestset "Stride" begin include("stride.jl") end
@safetestset "Layout" begin include("layout.jl") end
@safetestset "Static" begin include("static.jl") end
@safetestset "Engine" begin include("engine.jl") end

@testset "MoYeArray" begin
    @safetestset "MoYeArray" begin include("array.jl") end
    @safetestset "Broadcast" begin include("broadcast.jl") end
end

@safetestset "Tiling" begin include("tiling.jl") end

@testset "Device" begin
    @safetestset "Memory" begin include("device/memory.jl") end
    @safetestset "MMA" begin include("device/mmaop.jl") end
    @safetestset "MMATraits" begin include("device/mmatraits.jl") end
    @safetestset "Pointer" begin include("device/pointer.jl") endss
    @safetestset "LDMatrix" begin include("device/ldmatrix.jl") end
    @safetestset "Broadcast" begin include("device/broadcast.jl") end
end

@testset "Examples" begin
    @safetestset "Tiling MatMul" begin include("examples/tiling_matmul.jl") end
    @safetestset "Copy Async" begin include("examples/copy_async.jl") end
end
