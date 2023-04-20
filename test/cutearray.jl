using CuTe, Test

function test_alloc()
    slayout = @Layout (2,3)
    x = CuTeArray{Float32}(undef, slayout)
    fill!(x, 1.0f0)
    sum(x)
end

@test @allocated(test_alloc()) == 0
