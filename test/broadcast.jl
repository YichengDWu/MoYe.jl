using MoYe, Test

x = MoYeArray{Float32}(undef, @Layout((3,2)))
x2 = x .+ x
x3 = x .+ 1

@test x2 isa MoYeArray{Float32}
@test x2.layout == @Layout((3,2))
@test x2.engine isa ArrayEngine

@test x3 isa MoYeArray{Float32}
@test x3.layout == @Layout((3,2))
@test x3.engine isa ArrayEngine

y = MoYeArray{Float32}(undef, @Layout((3,), (2,)))
z = x .+ y
@test z.layout == x.layout
