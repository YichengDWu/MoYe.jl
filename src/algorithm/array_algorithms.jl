@inline function make_cutearray_like(::Type{T}, @nospecialize(layout::StaticLayout)) where {T <: Number}
    return CuTeArray{T}(undef, make_ordered_layout(layout)) # make the layout compact, hence not the same as `similar`
end
@inline function make_cutearray_like(::Type{T}, @nospecialize(x::CuTeArray)) where {T}
    return make_cutearray_like(T, layout(x))
end
@inline function make_cutearray_like(@nospecialize x::CuTeArray{T}) where {T}
    return make_cutearray_like(T, x)
end


@inline function make_fragment_like(::Type{T}, @nospecialize(layout::Layout)) where {T}
    return CuTeArray{T}(undef, make_fragment_like(layout))
end
@inline function make_fragment_like(::Type{T}, @nospecialize(x::CuTeArray)) where {T}
    return make_fragment_like(T, layout(x))
end
@inline function make_fragment_like(@nospecialize x::CuTeArray{T}) where {T}
    return make_fragment_like(T, x)
end

#  make_identity_tensor

# Layout manipulation, should return a non-owning CuTeArray
@inline flatten(@nospecialize(x::CuTeArray)) = CuTeArray(pointer(x), flatten(layout(x)))
@inline Base.coalesce(@nospecialize x::CuTeArray) = CuTeArray(pointer(x), coalesce(layout(x)))
@inline function Base.coalesce(@nospecialize(x::CuTeArray), @nospecialize(trg_profile::IntTuple))
    return CuTeArray(pointer(x), coalesce(layout(x), trg_profile))
end
@inline function group(@nospecialize(x::CuTeArray), B::IntType, E::IntType)
    return CuTeArray(pointer(x), group(layout(x), B, E))
end

# Algebra
@inline function logical_divide(@nospecialize(x::CuTeArray), tile::Tile)
    return CuTeArray(pointer(x), logical_divide(layout(x), tile))
end

@inline function zipped_divide(@nospecialize(x::CuTeArray), tile::Tile)
    return CuTeArray(pointer(x), zipped_divide(layout(x), tile))
end

@inline function tiled_divide(@nospecialize(x::CuTeArray), tile::Tile)
    return CuTeArray(pointer(x), tiled_divide(layout(x), tile))
end

"""
    local_partition(x::CuTeArray, tile::Tile, coord::Tuple)
    local_partition(x::CuTeArray, thread_layout::Layout, thread_id::Integer)

Partition a [`CuTeArray`](@ref) `x` into tiles that are parallised over.

## Examples

Say we have a [`CuTeArray`](@ref) `x` of shape `(6, 8)` and 4 threads of shape (2, 2). We would
like to  partition `x` with the 4 threads and get a view of the entries that the first thread
will work on. We can do this by calling `local_partition(x, (2, 2), 1)`.

```julia
julia> a = CuTeArray(pointer([i for i in 1:48]), @Layout((6,8)))
6×8 CuTeArray{Int64, 2, ViewEngine{Int64, Ptr{Int64}}, Layout{2, Tuple{Static.StaticInt{6}, Static.StaticInt{8}}, Tuple{Static.StaticInt{1}, Static.StaticInt{6}}}}:
 1   7  13  19  25  31  37  43
 2   8  14  20  26  32  38  44
 3   9  15  21  27  33  39  45
 4  10  16  22  28  34  40  46
 5  11  17  23  29  35  41  47
 6  12  18  24  30  36  42  48

julia> local_partition(a, (static(2), static(2)), (static(1), static(1)))
3×4 CuTeArray{Int64, 2, ViewEngine{Int64, Ptr{Int64}}, Layout{2, Tuple{Static.StaticInt{3}, Static.StaticInt{4}}, Tuple{Static.StaticInt{2}, Static.StaticInt{12}}}}:
 1  13  25  37
 3  15  27  39
 5  17  29  41
```

You can also use a thread layout and index with the thread id to get the tile:
```julia
julia> local_partition(a, @Layout((2,2), (1, 2)), 2)
3×4 CuTeArray{Int64, 2, ViewEngine{Int64, Ptr{Int64}}, Layout{2, Tuple{StaticInt{3}, StaticInt{4}}, Tuple{StaticInt{2}, StaticInt{12}}}}:
 2  14  26  38
 4  16  28  40
 6  18  30  42

julia> local_partition(a, @Layout((2,2), (2, 1)), 2)
3×4 CuTeArray{Int64, 2, ViewEngine{Int64, Ptr{Int64}}, Layout{2, Tuple{StaticInt{3}, StaticInt{4}}, Tuple{StaticInt{2}, StaticInt{12}}}}:
  7  19  31  43
  9  21  33  45
 11  23  35  47
```
"""
@inline function local_partition(@nospecialize(x::CuTeArray), tile::Tile, coord::Tuple)
    view(zipped_divide(x, tile), coord, repeat(:, rank(x)))
end
@inline function local_partition(@nospecialize(x::CuTeArray), tile::Layout, index::Integer)
    return local_partition(x, map(capacity, shape(tile)), get_congr_coord(tile, index))
end
@inline function local_partition(@nospecialize(x::CuTeArray), tile::Tile, coord, proj)
    return local_partition(x, dice(tile, proj), dice(coord, proj))
end
@inline function local_partition(@nospecialize(x::CuTeArray), tile::Layout, index::Integer, proj)
    return local_partition(x, dice(map(capacity, shape(tile)), proj), get_congr_coord(dice(tile, proj), index))
end

"""
    local_tile(@nospecialize(x::CuTeArray), tile::Tile, coord::Tuple)

Partition a [`CuTeArray`](@ref) `x` into tiles. This is similar to [`local_partition`](@ref) but
tiles are not parallelised.

```julia
julia> a = CuTeArray(pointer([i for i in 1:48]), @Layout((6,8)))

julia> local_tile(a, (static(2), static(2)), (static(1), static(1)))
2×2 CuTeArray{Int64, 2, ViewEngine{Int64, Ptr{Int64}}, Layout{2, Tuple{StaticInt{2}, StaticInt{2}}, Tuple{StaticInt{1}, StaticInt{6}}}}:
 1  7
 2  8
```
"""
@inline function local_tile(@nospecialize(x::CuTeArray), tile::Tile, coord::Tuple)
    R1 = length(tile)
    R2 = rank(x)
    return view(zipped_divide(x, tile), repeat(:, R1), append(coord, :, R2))
end
@inline function local_tile(@nospecialize(x::CuTeArray), tile::Tile, coord::Tuple, proj)
    return local_tile(x, dice(tile, proj), dice(coord, proj))
end



@inline function Base.fill!(x::CuTeArray{T, N, <:ArrayEngine}, val) where {T, N}
    b = ManualMemory.preserve_buffer(x)
    vb = ViewEngine(engine(x))
    GC.@preserve b begin
        for i in 1:length(vb)
            vb[i] = val
        end
    end
    return x
end

@inline function Base.sum(x::CuTeArray{T, N, <:ArrayEngine}) where {T, N}
    b = ManualMemory.preserve_buffer(x)
    vx = ViewEngine(engine(x))
    GC.@preserve b begin
        tmp = zero(T)
        for i in 1:length(vx)
            tmp += vx[i]
        end
        return tmp
    end
end

@inline zeros!(x::CuTeArray) = fill!(x, zero(eltype(x)))
