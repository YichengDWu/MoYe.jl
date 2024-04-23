using Test, SafeTestsets, CUDA

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
@safetestset "Copy" begin include("copy.jl") end

if CUDA.functional()
    @testset "Device" begin
        @safetestset "Memory" begin include("device/memory.jl") end
        @safetestset "MMA" begin include("device/mmaop.jl") end
        @safetestset "MMATraits" begin include("device/mmatraits.jl") end
        @safetestset "Pointer" begin include("device/pointer.jl") end
        @safetestset "LDMatrix" begin include("device/ldmatrix.jl") end
        @safetestset "Broadcast" begin include("device/broadcast.jl") end
        @safetestset "Tiled Copy" begin include("device/tiled_copy.jl") end
        @safetestset "MatMul" begin include("device/matmul.jl") end
    end
end

@testset "Host" begin
    @safetestset "CPU MatMul" begin include("host/tiling_matmul.jl") end
    @safetestset "Copy Async" begin include("host/copy_async.jl") end
end

@testset "Tiled MMA" begin
    @safetestset "Tiled MMA" begin include("tiled_mma.jl") end
end
