using MoYe, Test

A_data = collect(1:48*8*7)
B_data=collect(1:40*9*8)
C_data = collect(1:48*7*40*9)

A = MoYeArray(pointer(A_data), @Layout((48*7,8)))
B = MoYeArray(B_data, @Layout((40*9,8)))
C = MoYeArray(C_data, @Layout((48*7,40*9)))

# Tile size
tiled_mma = MoYe.make_tiled_mma(MMAOP_16x8x8_F16F16F16F16_TN(), @Layout((3,5)))

@test tile_size(tiled_mma, 1) == 16 * 3
@test tile_size(tiled_mma, 2) == 8 * 5
@test tile_size(tiled_mma, 3) == 8

thr_mma = MoYe.get_slice(tiled_mma, 1)

thr_C = partition_C(thr_mma, C)
thr_A = partition_A(thr_mma, A)
thr_B = partition_B(thr_mma, B)

@test size(thr_C, 2) == 7
@test size(thr_C, 3) == 9

@test size(thr_A, 2) == 7
@test size(thr_A, 3) == 1

@test size(thr_B, 2) == 9
@test size(thr_B, 3) == 1

frag_C = partition_fragment_C(thr_mma, C)
frag_A = partition_fragment_A(thr_mma, A)
frag_B = partition_fragment_B(thr_mma, B)

@test size(frag_C) == size(thr_C)
@test size(frag_A) == size(thr_A)
@test size(frag_B) == size(thr_B)
