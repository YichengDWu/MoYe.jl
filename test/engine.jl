using Moye, Test

function test_alloc()
    x = ArrayEngine{Float32}(one, static(10))
    @gc_preserve sum(x)
end

@test @allocated(test_alloc()) == 0
