using MoYe, Test, CUDA

p = reinterpret(Core.LLVMPtr{Int32, AS.Global}, 0)
p2 = reinterpret(Core.LLVMPtr{Int32, AS.Shared}, 0)

x = MoYeArray(p, @Layout(3));
x2 = MoYeArray(p2, @Layout(3));

@test isgmem(x) == true
@test isgmem(x2) == false
@test issmem(x) == false
@test issmem(x2) == true
