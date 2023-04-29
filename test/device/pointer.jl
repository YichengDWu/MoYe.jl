using Moye, Test, CUDA

p = reinterpret(Core.LLVMPtr{Int32, AS.Global}, 0)
p2 = reinterpret(Core.LLVMPtr{Int32, AS.Shared}, 0)

x = MoyeArray(p, @Layout(3));
x2 = MoyeArray(p2, @Layout(3));

@test isgmem(x) == true
@test isgmem(x2) == false
@test issmem(x) == false
@test issmem(x2) == true
