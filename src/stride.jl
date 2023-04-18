function coord_to_index0(coord::IntType, shape::IntType, stride::IntType)
    @inline
    return coord * stride
end
function coord_to_index0(coord::IntType, shape::Tuple{}, stride::Tuple{})
    @inline
    return zero(coord)
end
function coord_to_index0(coord::IntType, @nospecialize(shape::Tuple),
                         @nospecialize(stride::Tuple))
    s, d = first(shape), first(stride)
    q, r = divrem(coord, product(s))
    return coord_to_index0(r, s, d) +
           coord_to_index0(q, Base.tail(shape), Base.tail(stride))
end
function coord_to_index0(@nospecialize(coord::Tuple), @nospecialize(shape::Tuple),
                         @nospecialize(stride::Tuple))
    return flatsum(map(coord_to_index0, coord, shape, stride))
end

function coord_to_index(coord, shape, stride)
    return coord_to_index0(emap(Base.Fix2(-, static(1)), coord), shape, stride) + static(1)
end

# defaul stride, compact + column major
function coord_to_index0_horner(coord::IntType, shape::IntType)
    @inline
    return coord
end
function coord_to_index0_horner(coord::Tuple{}, shape::Tuple{})
    @inline
    return static(0)
end
function coord_to_index0_horner(@nospecialize(coord::Tuple), @nospecialize(shape::Tuple))
    c, s = first(coord), first(shape)
    return c + s * coord_to_index0_horner(Base.tail(coord), Base.tail(shape))
end

function coord_to_index0(coord, shape)
    iscongruent(coord, shape) ||
        throw(DimensionMismatch("coord and shape are not congruent"))
    return coord_to_index0_horner(flatten(coord), flatten(shape))
end

function coord_to_index(coord, shape)
    return coord_to_index0(emap(Base.Fix2(-, static(1)), coord), shape) + static(1)
end

abstract type AbstractMajor end
abstract type AbstractColMajor <: AbstractMajor end
abstract type AbstractRowMajor <: AbstractMajor end

struct CompactColMajor <: AbstractColMajor end
struct CompactRowMajor <: AbstractRowMajor end
const CompactMajor = Union{CompactColMajor, CompactRowMajor}

function compact_major(shape::IntType, current::IntType, major::Type{<:CompactMajor})
    return ifelse(isone(shape), zero(shape), current)
end
function compact_major(@nospecialize(shape::IntTuple), current::IntType,
                       major::Type{CompactColMajor})
    return tuple((compact_major(shape[i], current * product(shape[1:(i - 1)]), major) for i in 1:length(shape))...)
end

function compact_major(@nospecialize(shape::IntTuple), current::IntType,
                       major::Type{CompactRowMajor})
    return tuple((compact_major(shape[i], current * product(shape[(i + 1):(end - 1)]),
                                major) for i in 1:length(shape))...)
end

function compact_major(@nospecialize(shape::IntTuple), @nospecialize(current::IntTuple),
                       major::Type{<:CompactMajor})
    length(shape) == length(current) ||
        throw(DimensionMismatch("shape and current must have the same rank"))
    return tuple((compact_major(s, c, major) for (s, c) in zip(shape, current))...)
end

compact_col_major(shape, current=1) = compact_major(shape, current, CompactColMajor)
compact_row_major(shape, current=1) = compact_major(shape, current, CompactRowMajor)

### index_to_coord
function index_to_coord(index::IntType, shape::IntType, stride::IntType)
    @inline
    return ((index - one(index)) ÷ stride) % shape + one(index)
end
function index_to_coord(index::IntType, @nospecialize(shape::Tuple),
                        @nospecialize(stride::Tuple))
    length(shape) == length(stride) ||
        throw(DimensionMismatch("shape, and stride must have the same rank"))
    return tuple((index_to_coord(index, s, d) for (s, d) in zip(shape, stride))...)
end
function index_to_coord(index::IntType, @nospecialize(shape::Tuple), stride::IntType)
    return tuple((index_to_coord(index, s, d) for (s, d) in zip(shape,
                                                                compact_col_major(shape,
                                                                                  stride)))...)
end
function index_to_coord(@nospecialize(index::Tuple), @nospecialize(shape::Tuple),
                        stride::Tuple)
    length(index) == length(shape) == length(stride) ||
        throw(DimensionMismatch("index, shape, and stride must have the same rank"))
    return map(index_to_coord, index, shape, stride)
end

# default stride, compact + column major

function index_to_coord(index::IntType, shape::IntType)
    @inline
    return index
end
function index_to_coord(index::IntType, @nospecialize(shape::Tuple))
    return index_to_coord(index, shape, compact_col_major(shape, static(1)))
end
function index_to_coord(@nospecialize(index::Tuple), @nospecialize(shape::Tuple))
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

function compact_order(@nospecialize(shape::Tuple), @nospecialize(order::Tuple), org_shape,
                       org_order)
    return tuple((compact_order(s, o, org_shape, org_order) for (s, o) in zip(shape, order))...)
end
function compact_order(shape, order::IntType, @nospecialize(org_shape::Tuple),
                       @nospecialize(org_order::IntSequence))
    org_order = map(Base.Fix2(-, order), org_order)
    d = product(map((s, o) -> ifelse(signbit(o), product(s), static(1)), org_shape,
                       org_order))
    return compact_col_major(shape, d)
end
function compact_order(shape, order)
    iscongruent(shape, order) ||
        throw(DimensionMismatch("shape and order are not congruent"))
    return compact_order(shape, order, flatten(shape), flatten(order))
end
function compact_order(shape, major::CompactMajor)
    return compact_major(shape, static(1), major)
end
