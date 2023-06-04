function coord_to_index0(coord::IntType, shape::IntType, stride::IntType)
    @inline
    return coord * stride
end
function coord_to_index0(coord::Tuple{IntType}, shape::IntType, stride::IntType)
    @inline
    return first(coord) * stride
end
function coord_to_index0(coord::IntType, shape::Tuple{}, stride::Tuple{})
    @inline
    return zero(coord)
end
function coord_to_index0(coord::IntType, shape::Tuple, stride::Tuple)
    s, d = first(shape), first(stride)
    q, r = divrem(coord, product(s))
    return coord_to_index0(r, s, d) +
           coord_to_index0(q, Base.tail(shape), Base.tail(stride))
end
function coord_to_index0(coord::Tuple, shape::Tuple, stride::Tuple)
    return flatsum(map(coord_to_index0, coord, shape, stride))
end

@inline _offset(x::Colon) = Zero()
@inline _offset(x::Int) = x - one(x)
@inline _offset(x::StaticInt{N}) where {N} = StaticInt{N - 1}()
@inline _offset(x::NTuple{N, Colon}) where {N} = ntuple(Returns(Zero()), Val(N))
@inline function _offset(x::NTuple{N, Int}) where {N}
    return ntuple(Base.Fix2(-, 1) ∘ Base.Fix1(getindex, x), Val(N))
end
@inline _offset(x::Tuple) = map(_offset, x)

function coord_to_index(coord::IntType, shape, stride)
    idx = coord_to_index0(coord - one(coord), shape, stride)
    return idx + one(idx)
end
function coord_to_index(coord, shape, stride)
    idx = coord_to_index0(_offset(coord), shape, stride)
    return idx + one(idx)
end

# defaul stride, compact + column major
function coord_to_index0_horner(coord::IntType, shape::IntType)
    @inline
    return coord
end
function coord_to_index0_horner(coord::Tuple{}, shape::Tuple{})
    @inline
    return Zero()
end
Base.@assume_effects :total function coord_to_index0_horner(coord::Tuple, shape::Tuple)
    c, s = first(coord), first(shape)
    return c + s * coord_to_index0_horner(Base.tail(coord), Base.tail(shape))
end

function coord_to_index0(coord, shape)
  #  iscongruent(coord, shape) ||
  #      throw(DimensionMismatch("coord and shape are not congruent"))
    return coord_to_index0_horner(flatten(coord), flatten(shape))
end

function coord_to_index(coord::IntType, shape)
    return coord_to_index0(coord - one(coord), shape) + one(coord)
end
function coord_to_index(coord, shape)
    idx = coord_to_index0(_offset(coord), shape)
    return idx + one(idx)
end

function index_to_coord(index::IntType, shape::StaticInt{1}, stride::IntType)
    return Zero()
end
function index_to_coord(index::IntType, shape::IntType, stride::IntType)
    crd = ((index - one(index)) ÷ stride) % shape
    return crd + one(crd)
end
function index_to_coord(index::IntType, shape::Tuple, stride::Tuple)
    length(shape) == length(stride) ||
        throw(DimensionMismatch("shape, and stride must have the same rank"))
    return let index = index
        map((s, d) -> index_to_coord(index, s, d), shape, stride)
    end
end
function index_to_coord(index::IntType, shape::Tuple, stride::IntType)
    return let index = index
        map((s, d) -> index_to_coord(index, s, d), shape, compact_col_major(shape, stride))
    end
end
function index_to_coord(index::Tuple, shape::Tuple, stride::Tuple)
    length(index) == length(shape) == length(stride) ||
        throw(DimensionMismatch("index, shape, and stride must have the same rank"))
    return map(index_to_coord, index, shape, stride)
end

# default stride, compact + column major
function index_to_coord(index::IntType, shape::IntType)
    @inline
    return index
end
function index_to_coord(index::IntType, shape::Tuple)
    return index_to_coord(index, shape, compact_col_major(shape, One()))
end
function index_to_coord(index::Tuple, shape::Tuple)
    length(index) == length(shape) ||
        throw(DimensionMismatch("index and shape must have the same rank"))
    return map(index_to_coord, index, shape)
end

"""
Transoform a coordinate in one shape to a coordinate in another shape.
"""
function coord_to_coord(@nospecialize(coord::Tuple), @nospecialize(src_shape::Tuple),
                        @nospecialize(dst_shape::Tuple))
    length(coord) == length(src_shape) == length(dst_shape) ||
        throw(DimensionMismatch("coord, shape1, and shape2 must have the same rank"))
    return map(coord_to_coord, coord, src_shape, dst_shape)
end
function coord_to_coord(coord, src_shape, dst_shape)
    return index_to_coord(coord_to_index(coord, src_shape), dst_shape)
end

struct LayoutLeft end
struct LayoutRight end

const GenColMajor = LayoutLeft
const GenRowMajor = LayoutRight

struct CompactLambda{Major} end

Base.@assume_effects :total function compact(shape::Tuple, current::IntType, ::Type{LayoutLeft})
    return _foldl(CompactLambda{LayoutLeft}(), shape, ((), current))
end
Base.@assume_effects :total function compact(shape::Tuple, current::IntType, ::Type{LayoutRight})
    return _foldl(CompactLambda{LayoutRight}(), reverse(shape), ((), current))
end
function compact(shape::StaticInt{1}, current::StaticInt, ::Type{Major}) where {Major}
    @inline
    return (Zero(), current)
end
function compact(shape::StaticInt{1}, current::Integer, ::Type{Major}) where {Major}
    @inline
    return (Zero(), current)
end
@generated function compact(shape::StaticInt{N}, current::StaticInt{M},
                            ::Type{Major}) where {Major, N, M}
    return :((current, $(StaticInt{N * M}())))
end
function compact(shape::IntType, current::IntType, ::Type{Major}) where {Major}
    @inline
    return (current, current * shape)
end

function compact_major(shape::Tuple, current::Tuple, major::Type{Major}) where {Major}
    length(shape) == length(current) ||
        throw(DimensionMismatch("shape and current must have the same rank"))
    return map((s, c) -> compact_major(s, c, major), shape, current)
end
function compact_major(shape, current::IntType, major::Type{Major}) where {Major}
    return first(compact(shape, current, major))
end

Base.@assume_effects :total function (::CompactLambda{LayoutLeft})(init, si)
    result = compact(si, init[2], LayoutLeft)
    return (append(init[1], result[1]), result[2])
end
Base.@assume_effects :total function (::CompactLambda{LayoutRight})(init, si)
    result = compact(si, init[2], LayoutRight)
    return (prepend(init[1], result[1]), result[2])
end

compact_col_major(shape, current=One()) = compact_major(shape, current, LayoutLeft)
compact_row_major(shape, current=One()) = compact_major(shape, current, LayoutRight)

function compact_order(shape::Tuple, order::Tuple, old_shape, old_order)
    return let old_shape = old_shape, old_order = old_order
        map((x, y) -> compact_order(x, y, old_shape, old_order), shape, order)
    end
end
function compact_order(shape, order::IntType, old_shape, old_order)
    d = let order = order
        product(map((s, o) -> ifelse(o < order, product(s), One()), old_shape, old_order))
    end
    return compact_col_major(shape, d)
end
function compact_order(shape, order)
    return compact_order(shape, order, tuple(flatten(shape)...), tuple(flatten(order)...))
end
function compact_order(shape, ::Type{LayoutLeft})
    return compact_col_major(shape)
end
function compact_order(shape, ::Type{LayoutRight})
    return compact_row_major(shape)
end
