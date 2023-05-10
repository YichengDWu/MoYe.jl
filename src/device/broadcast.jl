using MoYe, CUDA

function f()
    a = MoYeArray{Float64}(undef, @Layout((3,2)))
    fill!(a, one(eltype(a)))
    a .= a .* 2
    @cushow sum(a)
    b = CUDA.exp.(a)
    @cushow sum(b)
    return nothing
end

if CUDA.functional()
    @cuda f()
end
