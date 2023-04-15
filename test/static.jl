using CuTe, Test, Static

layout = make_layout(static((3, 4)), static((1, 2)))
@test sizeof(layout) == 0
