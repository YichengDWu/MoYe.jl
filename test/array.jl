using MoYe, Test

function test_alloc()
    slayout = @Layout (2, 3)
    x = MoYeArray{Float32}(undef, slayout)
    fill!(x, 1.0f0)
    return sum(x)
end

@test @allocated(test_alloc()) == 0

@testset "Constructors" begin
    @test_nowarn MoYeArray{Float32}(undef, static((2, 3)))
    @test_throws MethodError MoYeArray{Float32}(undef, (2, 3))

    @test_nowarn MoYeArray{Float32}(undef, static((2, 3)), GenRowMajor)
    @test_nowarn MoYeArray{Float32}(undef, static((2, 3)), GenColMajor)

    A = rand(3)
    ca = MoYeArray(pointer(A), static((3, 1)))
    ca2 = MoYeArray(pointer(A), static((3, 1)), GenRowMajor)
    @test ca.engine isa ViewEngine
    @test ca2.engine isa ViewEngine
end

@testset "Static indexing" begin
    A = MoYeArray{Float64}(undef, @Layout((3,4)))
    @test A[_1] == A[1]
    A[_1] = 1.0
    @test A[_1] == 1.0

    data = rand(3,4)
    B = MoYeArray(pointer(data), @Layout((3,4)))
    @test B[_1] == B[1]
    B[_1] = 1.0
    @test B[_1] == 1.0
end

@testset "Array Operations" begin
    @testset "View" begin
        ca = MoYeArray{Float32}(undef, static((2, 3)))
        va = view(ca, :, 1)
        @test va isa MoYeArray
        @test va.engine isa ViewEngine
        @test va.layout.shape == tuple(_2)

        va2 = view(ca, :, :)
        @test va2 isa MoYeArray
        @test va2.engine isa ViewEngine
        @test va2.layout.shape == tuple(_2, _3)
    end

    @testset "Copy" begin
        ca = MoYeArray{Float32}(undef, static((2, 3)))
        zeros!(ca)
        ca2 = copy(ca)
        @test ca2 isa MoYeArray
        @test ca2.engine isa ArrayEngine
        @test ca2 == ca

        A = ones(6)
        ca3 = MoYeArray(pointer(A), _6)
        ca4 = copy(ca3)
        @test ca4 isa MoYeArray
        @test ca4.engine isa ArrayEngine
        @test ca4 == ca3
    end

    @testset "similar" begin
        ca = MoYeArray{Float32}(undef, static((2, 3)))
        ca2 = similar(ca)
        @test ca2 isa MoYeArray
        @test ca2.engine isa ArrayEngine
        @test ca2.layout == ca.layout

        A = ones(6)
        ca3 = MoYeArray(pointer(A), _6)
        ca4 = similar(ca3)
        @test ca4 isa MoYeArray
        @test ca4.engine isa ArrayEngine
        @test ca4.layout == ca3.layout
    end

    @testset "recast" begin
        data = collect(1:12)
        a = MoYeArray(pointer(data), @Layout((4,3)))

        @testset "downcast" begin
            b = recast(Int32, a)
            @test b.layout == @Layout((8,3))
            @test b.engine isa ViewEngine

            @testset "dynamic layout" begin
                a′ = MoYeArray(pointer(data), make_layout((4,3)))
                b′ = recast(Int32, a′)
                @test b′.layout == make_layout((8,3))
                @test b′.engine isa ViewEngine
            end
        end

        @testset "upcast" begin
            c = recast(UInt128, a)
            @test c.layout == @Layout((2,3))
            @test c.engine isa ViewEngine
        end
    end
end

@testset "BLAS" begin
    @testset "fill! and sum" begin
        ca = MoYeArray{Float32}(undef, static((2, 3)))
        fill!(ca, 1.0f0)
        @test all(ca .== 1.0f0)
        @test sum(ca) == 6.0f0
    end
end

@testset "Unsqueeze" begin
    a = MoYeArray{Float32}(undef, static((2, 3)))
    b = MoYe.append_dim(a, _3);
    @test b.layout == @Layout (2, 3, 1) (1,2,0)

    c = MoYe.prepend_dim(a, _3);
    @test c.layout == @Layout (1, 2, 3) (0,1,2)
end

@testset "Recast" begin
    x = MoYeArray{Float32}(undef, static((4, 3)))
    zeros!(x)
    @testset "Upcast" begin
        x2 = recast(Float64, x)
        @test x2 isa MoYeArray{Float64}
        @test x2.layout == @Layout (2, 3)
        @test x == recast(Float32, x2)
    end

    @testset "Downcast" begin
        x2 = recast(Float16, x)
        zeros!(x2)
        @test x2 isa MoYeArray{Float16}
        @test x2.layout == @Layout (8, 3)
        @test x == recast(Float32, x2)
    end
end
