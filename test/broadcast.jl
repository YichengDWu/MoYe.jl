using Moye, Test

x = MoyeArray{Float32}(undef, @Layout((3,2)))
x2 = x .+ x
x3 = x .+ 1

@test x2 isa MoyeArray{Float32}
@test x2.layout == @Layout((3,2))
@test x2.engine isa ArrayEngine

@test x3 isa MoyeArray{Float32}
@test x3.layout == @Layout((3,2))
@test x3.engine isa ArrayEngine
