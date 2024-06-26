@inline function make_fragment_like(::Type{T}, layout::Layout) where {T}
    return MoYeArray{T}(undef, make_fragment_like(layout))
end
@inline function make_fragment_like(::Type{T}, x::MoYeArray) where {T}
    return make_fragment_like(T, layout(x))
end
@inline function make_fragment_like(x::MoYeArray{T}) where {T}
    return make_fragment_like(T, x)
end

#  make_identity_tensor

# Layout manipulation, should return a non-owning MoYeArray
@inline flatten(@nospecialize(x::MoYeArray)) = MoYeArray(pointer(x), flatten(layout(x)))
@inline Base.coalesce(@nospecialize x::MoYeArray) = MoYeArray(pointer(x), coalesce(layout(x)))
@inline function Base.coalesce(@nospecialize(x::MoYeArray), @nospecialize(trg_profile::IntTuple))
    return MoYeArray(pointer(x), coalesce(layout(x), trg_profile))
end
@inline function group_modes(@nospecialize(x::MoYeArray), B::IntType, E::IntType)
    return MoYeArray(pointer(x), group(layout(x), B, E))
end

# Algebra
@inline function logical_divide(@nospecialize(x::MoYeArray), tile)
    return MoYeArray(pointer(x), logical_divide(layout(x), tile))
end

@inline function zipped_divide(@nospecialize(x::MoYeArray), tile)
    return MoYeArray(pointer(x), zipped_divide(layout(x), tile))
end

@inline function tiled_divide(@nospecialize(x::MoYeArray), tile)
    return MoYeArray(pointer(x), tiled_divide(layout(x), tile))
end

@inline function local_partition(x::MoYeArray{T,N}, tile::Tile, coord::Tuple) where {T,N}
    return view(zipped_divide(x, tile), coord, ntuple(i -> Colon(), Val(N)))
end
@inline function local_partition(@nospecialize(x::MoYeArray), tile::Layout, index::DInt)
    return local_partition(x, map(capacity, shape(tile)), get_congr_coord(tile, index))
end
@inline function local_partition(@nospecialize(x::MoYeArray), tile::Tile, coord, proj)
    return local_partition(x, dice(tile, proj), dice(coord, proj))
end
@inline function local_partition(@nospecialize(x::MoYeArray), tile::Layout, index::Integer, proj)
    return local_partition(x, dice(map(capacity, shape(tile)), proj), get_congr_coord(dice(tile, proj), index))
end

function composition(x::MoYeArray, l)
    @inline
    return MoYeArray(pointer(x), composition(layout(x), l))
end
function Base.:(∘)(x::MoYeArray, l)
    @inline
    return MoYeArray(pointer(x), composition(layout(x), l))
end

"""
    @parallelize x::MoYeArray threadgroup_layout::Layout thread_idx::Int

Partition `x` with `size(threadgroup_layout)` threads, and return the view of the entries that the thread at `thread_idx` will work on.

## Examples

Say we have a [`MoYeArray`](@ref) `x` of shape `(6, 8)` and 4 threads of shape (2, 2). We would
like to  partition `x` with the 4 threads and get a view of the entries that the first thread
will work on. We can do this by calling `@parallelize(x, (2, 2), 1)`.

```julia
julia> a = MoYeArray(pointer([i for i in 1:48]), @Layout((6,8)))
6×8 MoYeArray{Int64, 2, ViewEngine{Int64, Ptr{Int64}}, Layout{2, Tuple{Static.StaticInt{6}, Static.StaticInt{8}}, Tuple{Static.StaticInt{1}, Static.StaticInt{6}}}}:
 1   7  13  19  25  31  37  43
 2   8  14  20  26  32  38  44
 3   9  15  21  27  33  39  45
 4  10  16  22  28  34  40  46
 5  11  17  23  29  35  41  47
 6  12  18  24  30  36  42  48

julia> @parallelize a (_2, _2) (1, 1)
3×4 MoYeArray{Int64, 2, ViewEngine{Int64, Ptr{Int64}}, Layout{2, Tuple{Static.StaticInt{3}, Static.StaticInt{4}}, Tuple{Static.StaticInt{2}, Static.StaticInt{12}}}}:
 1  13  25  37
 3  15  27  39
 5  17  29  41
```

You can also pass in a thread layout and a thread id to get the tile:
```julia
julia> @parallelize a @Layout((2,2), (1, 2)) 2
3×4 MoYeArray{Int64, 2, ViewEngine{Int64, Ptr{Int64}}, Layout{2, Tuple{StaticInt{3}, StaticInt{4}}, Tuple{StaticInt{2}, StaticInt{12}}}}:
 2  14  26  38
 4  16  28  40
 6  18  30  42

julia> @parallelize a @Layout((2,2), (2, 1)) 2
3×4 MoYeArray{Int64, 2, ViewEngine{Int64, Ptr{Int64}}, Layout{2, Tuple{StaticInt{3}, StaticInt{4}}, Tuple{StaticInt{2}, StaticInt{12}}}}:
  7  19  31  43
  9  21  33  45
 11  23  35  47
```
"""
macro parallelize(x, tile, coord, proj...)
    if length(proj) == 0
        return quote
            local_partition($(esc(x)), static($(esc(tile))), $(esc(coord)))
        end
    else
        return quote
            local_partition($(esc(x)), static($(esc(tile))), $(esc(coord)), static($(esc(proj[1]))))
        end
    end
end

@inline function local_tile(x::MoYeArray, tile::Tile, coord::Tuple)
    R1 = length(tile)
    R2 = rank(x)
    return view(zipped_divide(x, tile), ntuple(i -> :, R1), append(coord, :, StaticInt{R2}()))
end
@inline function local_tile(x::MoYeArray, tile::Tile, coord::Tuple, proj)
    return local_tile(x, dice(tile, proj), dice(coord, proj))
end

"""
    @tile x::MoYeArray threadgroup_shape::Tile threadgroup_coord::Tuple

Partition `x` with `threadgroup_shape`. Return the view of the entries of `x` that the thread group at `threadgroup_coord` will work on.

## Examples

```julia
julia> a = MoYeArray(pointer([i for i in 1:48]), @Layout((6,8)))
6×8 MoYeArray{Int64, 2, ViewEngine{Int64, Ptr{Int64}}, Layout{2, Tuple{Static.StaticInt{6}, Static.StaticInt{8}}, Tuple{Static.StaticInt{1}, Static.StaticInt{6}}}} with indices _1:_6×_1:_8:
 1   7  13  19  25  31  37  43
 2   8  14  20  26  32  38  44
 3   9  15  21  27  33  39  45
 4  10  16  22  28  34  40  46
 5  11  17  23  29  35  41  47
 6  12  18  24  30  36  42  48

julia> @tile a (_2, _2) (1, 1)
2×2 MoYeArray{Int64, 2, ViewEngine{Int64, Ptr{Int64}}, Layout{2, Tuple{StaticInt{2}, StaticInt{2}}, Tuple{StaticInt{1}, StaticInt{6}}}}:
 1  7
 2  8

```
"""
macro tile(x, tile, coord, proj...)
    if length(proj) == 0
        return quote
            local_tile($(esc(x)), static($(esc(tile))), $(esc(coord)))
        end
    else
        return quote
            local_tile($(esc(x)), static($(esc(tile))), $(esc(coord)), static($(esc(proj[1]))))
        end
    end
end

@inline function Base.fill!(x::NonOwningArray, val)
    vx = engine(x)
    @loopinfo unroll for i in eachindex(x)
        @inbounds vx[i] = val
    end
    return x
end
@inline Base.fill!(x::OwningArray, val) = @gc_preserve fill!(x, val)

@inline function Base.sum(x::NonOwningArray{T}) where T
    vx = engine(x)
    tmp = zero(T)
    @loopinfo unroll for i in 1:length(x)
        @inbounds tmp += vx[i]
    end
    return tmp
end
@inline Base.sum(x::OwningArray) = @gc_preserve sum(x)

"""
    zeros!(x::MoYeArray)

Fill `x` with zeros.
"""
@inline zeros!(x::MoYeArray) = fill!(x, zero(eltype(x)))

function max_common_vector(src::MoYeArray{TS}, dst::MoYeArray{TD}) where {TS, TD}
    if sizeof(TS) == sizeof(TD) && isbitstype(TS) && isbitstype(TD)
        return max_common_vector(src.layout, dst.layout)
    else
        return Zero()
    end
end

@device_override function foreach(f::F, x::MoYeArray) where {F}
    @loopinfo unroll for i in eachindex(x)
        f(x[i])
    end
    return nothing
end

@device_override function map!(f::F, x::MoYeArray, y::MoYeArray) where {F}
    @loopinfo unroll for i in eachindex(x)
        x[i] = f(y[i])
    end
    return nothing
end
@device_override @inline function map!(f::F, x::MoYeArray) where {F}
    @loopinfo unroll for i in eachindex(x)
        x[i] = f(x[i])
    end
    return nothing
end
