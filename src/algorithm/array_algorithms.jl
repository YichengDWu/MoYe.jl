@inline function make_cutearray_like(::Type{T}, layout::StaticLayout) where {T<:Number}
    return CuTeArray{T}(make_ordered_layout(layout)) # make the layout compact, hence not the same as `similar`
end
@inline function make_cutearray_like(::Type{T}, x::CuTeArray) where {T}
    return make_cutearray_like(T, layout(x))
end
@inline function make_cutearray_like(x::CuTeArray{T}) where {T}
    return make_cutearray_like(T, x)
end

@inline function make_fragment_like(::Type{T}, layout::Layout) where {T}
    return CuTeArray{T}(make_fragment_like(layout))
end
@inline function make_fragment_like(::Type{T}, ::CuTeArray) where {T}
    return make_fragment_like(T, layout(x))
end
@inline function make_fragment_like(x::CuTeArray{T}) where {T}
    return make_fragment_like(T, x)
end

#  make_identity_tensor


# Layout manipulation, should return a non-owning CuTeArray
@inline flatten(x::CuTeArray) = CuTeArray(pointer(x), flatten(layout(x)))
@inline Base.coalesce(x::CuTeArray) = CuTeArray(pointer(x), coalesce(layout(x)))
@inline Base.coalesce(x::CuTeArray, @nospecialize trg_profile::IntTuple) = CuTeArray(pointer(x), coalesce(layout(x), trg_profile))
@inline group(x::CuTeArray, B::IntType, E::IntType) = CuTeArray(pointer(x), group(layout(x), B, E))


# Algebra
@inline function logical_divide(x::CuTeArray, tile::Tile)
    return CuTeArray(pointer(x), logical_divide(layout(x), tile))
end

@inline function zipped_divide(x::CuTeArray, tile::Tile)
    return CuTeArray(pointer(x), zipped_divide(layout(x), tile))
end

@inline function tiled_divide(x::CuTeArray, tile::Tile)
    return CuTeArray(pointer(x), tiled_divide(layout(x), tile))
end

#local_partition


@inline function local_tile(x::CuTeArray, tile::Tile, coord::IntTuple)
    R1 = length(tile)
    R2 = rank(x)
    return zipped_divide(x, tile)(repeat(:, R1), append(coord, :, R2))
end
@inline function local_tile(x::CuTeArray, tile::Tile, coord::IntTuple, proj)
    return local_tile(x, dice(tile, proj), dice(coord, proj))
end

# Array operations
@inline Base.similar(x::CuTeArray{T}) where {T} = similar(x, T)
@inline Base.similar(x::CuTeArray{S,N,<:ArrayEngine}, ::Type{T}) where {S,N,T} = CuTeArray{T}(undef, layout(x))

@inline function Base.fill!(x::CuTeArray{T,N,<:ArrayEngine}, val) where {T,N}
    b = ManualMemory.preserve_buffer(x)
    GC.@preserve b begin
        fill!(ViewEngine(engine(x)), val)
    end
    return x
end

@inline function Base.sum(x::CuTeArray{T,N,<:ArrayEngine}) where {T,N}
    b = ManualMemory.preserve_buffer(x)
    GC.@preserve b begin
        sum(ViewEngine(engine(x)))
    end
end
