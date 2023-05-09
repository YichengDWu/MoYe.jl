# MoYe

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://YichengDWu.github.io/MoYe.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://YichengDWu.github.io/MoYe.jl/dev/)
[![Build Status](https://github.com/YichengDWu/MoYe.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/YichengDWu/MoYe.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/YichengDWu/MoYe.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/YichengDWu/MoYe.jl)

The `MoYe.jl` library draws significant inspiration from NVIDIA's [CuTe](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/00_quickstart.md) and is built with similar underlying structures. I believe the credit goes to @ccecka for creating CuTe.

The name **Mo Ye** is derived from an ancient Chinese [legend of swordsmiths](https://en.wikipedia.org/wiki/Gan_Jiang_and_Mo_Ye).

## Installation
```julia
pkg> add MoYe
```

## Quick Start
``julia
julia> a = MoYeArray(pointer([i for i in 1:48]), @Layout((6,8)))
6×8 MoYeArray{Int64, 2, ViewEngine{Int64, Ptr{Int64}}, Layout{2, Tuple{Static.StaticInt{6}, Static.StaticInt{8}}, Tuple{Static.StaticInt{1}, Static.StaticInt{6}}}}:
 1   7  13  19  25  31  37  43
 2   8  14  20  26  32  38  44
 3   9  15  21  27  33  39  45
 4  10  16  22  28  34  40  46
 5  11  17  23  29  35  41  47
 6  12  18  24  30  36  42  48

julia> subtile_a = @tile a static((3,4)) (1, 2)
3×4 MoYeArray{Int64, 2, ViewEngine{Int64, Ptr{Int64}}, Layout{2, Tuple{Static.StaticInt{3}, Static.StaticInt{4}}, Tuple{Static.StaticInt{1}, Static.StaticInt{6}}}}:
 25  31  37  43
 26  32  38  44
 27  33  39  45

julia> workitems_a = @parallelize subtile_a static((3,2)) (1,1)
1×2 MoYeArray{Int64, 2, ViewEngine{Int64, Ptr{Int64}}, Layout{2, Tuple{Static.StaticInt{1}, Static.StaticInt{2}}, Tuple{Static.StaticInt{0}, Static.StaticInt{12}}}}:
 25  37

julia> for i in eachindex(workitems_a)
           workitems_a[i] = 0
       end

julia> a
6×8 MoYeArray{Int64, 2, ViewEngine{Int64, Ptr{Int64}}, Layout{2, Tuple{Static.StaticInt{6}, Static.StaticInt{8}}, Tuple{Static.StaticInt{1}, Static.StaticInt{6}}}}:
 1   7  13  19   0  31   0  43
 2   8  14  20  26  32  38  44
 3   9  15  21  27  33  39  45
 4  10  16  22  28  34  40  46
 5  11  17  23  29  35  41  47
 6  12  18  24  30  36  42  48
 ```
