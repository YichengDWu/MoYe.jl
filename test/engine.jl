using CuTe, Test

function test_alloc()
    x = ArrayEngine{Float32}(one, static(10))
    GC.@preserve x begin sum(ViewEngine(x)) end
end

@test @allocated(test_alloc()) == 0
