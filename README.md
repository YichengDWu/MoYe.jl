# MoYe

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://YichengDWu.github.io/MoYe.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://YichengDWu.github.io/MoYe.jl/dev/)
[![Build Status](https://github.com/YichengDWu/MoYe.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/YichengDWu/MoYe.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/YichengDWu/MoYe.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/YichengDWu/MoYe.jl)

The `MoYe.jl` library draws significant inspiration from NVIDIA's [CuTe](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/00_quickstart.md) and is built with similar underlying structures.

The name **Mo Ye** is derived from an ancient Chinese [legend of swordsmiths](https://en.wikipedia.org/wiki/Gan_Jiang_and_Mo_Ye).

## Installation
```julia
pkg> add MoYe
```

## Quick Start
```julia
julia> data = [i for i in 1:48];

julia> a = MoYeArray(data, @Layout((6,8)))
6×8 MoYeArray{Int64, 2, ViewEngine{Int64, Ptr{Int64}}, Layout{2, Tuple{Static.StaticInt{6}, Static.StaticInt{8}}, Tuple{Static.StaticInt{1}, Static.StaticInt{6}}}} with indices _1:_6×_1:_8:
 1   7  13  19  25  31  37  43
 2   8  14  20  26  32  38  44
 3   9  15  21  27  33  39  45
 4  10  16  22  28  34  40  46
 5  11  17  23  29  35  41  47
 6  12  18  24  30  36  42  48

julia> subtile_a = @tile a (_3, _4) (1, 2)
3×4 MoYeArray{Int64, 2, ViewEngine{Int64, Ptr{Int64}}, Layout{2, Tuple{Static.StaticInt{3}, Static.StaticInt{4}}, Tuple{Static.StaticInt{1}, Static.StaticInt{6}}}} with indices _1:_3×_1:_4:
 25  31  37  43
 26  32  38  44
 27  33  39  45

julia> workitems_a = @parallelize subtile_a (_3, _2) (1, 1)
1×2 MoYeArray{Int64, 2, ViewEngine{Int64, Ptr{Int64}}, Layout{2, Tuple{Static.StaticInt{1}, Static.StaticInt{2}}, Tuple{Static.StaticInt{0}, Static.StaticInt{12}}}} with indices _1:_1×_1:_2:
 25  37

julia> for i in eachindex(workitems_a)
                  workitems_a[i] = 0
              end

julia> a
6×8 MoYeArray{Int64, 2, ViewEngine{Int64, Ptr{Int64}}, Layout{2, Tuple{Static.StaticInt{6}, Static.StaticInt{8}}, Tuple{Static.StaticInt{1}, Static.StaticInt{6}}}} with indices _1:_6×_1:_8:
 1   7  13  19   0  31   0  43
 2   8  14  20  26  32  38  44
 3   9  15  21  27  33  39  45
 4  10  16  22  28  34  40  46
 5  11  17  23  29  35  41  47
 6  12  18  24  30  36  42  48

julia> @tile subtile_a (_3, _1) (1, 2)
3×1 MoYeArray{Int64, 2, ViewEngine{Int64, Ptr{Int64}}, Layout{2, Tuple{Static.StaticInt{3}, Static.StaticInt{1}}, Tuple{Static.StaticInt{1}, Static.StaticInt{0}}}} with indices _1:_3×_1:_1:
 31
 32
 33
 ```
 
## Tile Iterator

```julia
julia> data = collect(1:36);

julia> A = MoYeArray(data, @Layout((4,9)))
4×9 MoYeArray{Int64, 2, ViewEngine{Int64, Ptr{Int64}}, Layout{2, Tuple{Static.StaticInt{4}, Static.StaticInt{9}}, Tuple{Static.StaticInt{1}, Static.StaticInt{4}}}} with indices _1:_4×_1:_9:
 1  5   9  13  17  21  25  29  33
 2  6  10  14  18  22  26  30  34
 3  7  11  15  19  23  27  31  35
 4  8  12  16  20  24  28  32  36

julia> tiled_A = zipped_divide(A, (@Layout(2), @Layout(3))) 
6×6 MoYeArray{Int64, 2, ViewEngine{Int64, Ptr{Int64}}, Layout{2, Tuple{Tuple{Static.StaticInt{2}, Static.StaticInt{3}}, Tuple{Static.StaticInt{2}, Static.StaticInt{3}}}, Tuple{Tuple{Static.StaticInt{1}, Static.StaticInt{4}}, Tuple{Static.StaticInt{2}, Static.StaticInt{12}}}}} with indices _1:_6×_1:_6:
  1   3  13  15  25  27
  2   4  14  16  26  28
  5   7  17  19  29  31
  6   8  18  20  30  32
  9  11  21  23  33  35
 10  12  22  24  34  36

julia> for i in axes(tiled_A, 2)
                  @show view(tiled_A, :, i)
              end
view(tiled_A, :, i) = [1, 2, 5, 6, 9, 10]
view(tiled_A, :, i) = [3, 4, 7, 8, 11, 12]
view(tiled_A, :, i) = [13, 14, 17, 18, 21, 22]
view(tiled_A, :, i) = [15, 16, 19, 20, 23, 24]
view(tiled_A, :, i) = [25, 26, 29, 30, 33, 34]
view(tiled_A, :, i) = [27, 28, 31, 32, 35, 36]
```
 # Current Status
 
 `Tensor Core MMA`: High-level programming on tensor cores has been implemented, as shown in the
 [example](https://github.com/YichengDWu/MoYe.jl/blob/main/examples/gemm_tiled.jl) file. Integration with `ldmatrix` is coming soon.

Contributions from the community are very much welcome and encouraged. 

