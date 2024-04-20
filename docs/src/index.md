```@meta
CurrentModule = MoYe
```

# MoYe

Documentation for [MoYe](https://github.com/YichengDWu/MoYe.jl).

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

```@repl
using MoYe 
data = collect(1:36);
A = MoYeArray(data, @Layout((4,9)))
tiled_A = zipped_divide(A, (@Layout(2), @Layout(3))) # 2 × 3 tile
for i in axes(tiled_A, 2)
    @show view(tiled_A, :, i)
end
```
