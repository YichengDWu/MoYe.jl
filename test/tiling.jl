using MoYe, Test

a = MoYeArray(pointer([i for i in 1:48]), @Layout((6,8)))
tile = @tile a (_3, _2) (1, :);
@test rank(tile.layout) == 3
