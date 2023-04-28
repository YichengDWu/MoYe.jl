using Shambles, Test, Static

layout = make_layout(static((3, 4)), static((1, 2)))
@test sizeof(layout) == 0
@test size(layout) isa StaticInt
@test cosize(layout) isa StaticInt
@test depth(layout) isa StaticInt
@test layout(One()) isa StaticInt
@test layout(static((1, 2))) isa StaticInt
