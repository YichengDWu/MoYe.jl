function coord_to_index0(coord::Int, shape::Int, stride::Int)
    @inline
    return coord * stride
end
function coord_to_index0(coord::Int, shape::Tuple{}, stride::Tuple{})
    @inline
    return zero(coord)
end
function coord_to_index0(coord::Int, shape::Tuple, stride::Tuple)
    s, d = first(shape), first(stride)
    q, r = divrem(coord, product(s))
    return coord_to_index0(r, s, d) +
           coord_to_index0(q, Base.tail(shape), Base.tail(stride))
end
function coord_to_index0(coord::Tuple, shape::Tuple, stride::Tuple)
    return flatsum(map(coord_to_index0, coord, shape, stride))
end

function coord_to_index(coord, shape, stride)
    return coord_to_index0(emap(Base.Fix2(-, 1), coord), shape, stride) + 1
end

# defaul stride, compact + column major
function coord_to_index0_horner(coord::Int, shape::Int)
    @inline
    return coord
end
function coord_to_index0_horner(coord::Tuple{}, shape::Tuple{})
    @inline
    return 0
end
function coord_to_index0_horner(coord::Tuple, shape::Tuple)
    c, s = first(coord), first(shape)
    return c + s * coord_to_index0_horner(Base.tail(coord), Base.tail(shape))
end

function coord_to_index0(coord, shape)
    iscongruent(coord, shape) ||
        throw(DimensionMismatch("coord and shape are not congruent"))
    return coord_to_index0_horner(flatten(coord), flatten(shape))
end

function coord_to_index(coord, shape)
    return coord_to_index0(emap(Base.Fix2(-, 1), coord), shape) + 1
end

abstract type AbstractMajor end
abstract type AbstractColMajor <: AbstractMajor end
abstract type AbstractRowMajor <: AbstractMajor end

struct CompactColMajor <: AbstractColMajor end
struct CompactRowMajor <: AbstractRowMajor end
const CompactMajor = Union{CompactColMajor, CompactRowMajor}

const GenColMajor = CompactColMajor()
const GenRowMajor = CompactRowMajor()

function compact_major(shape::Int, current::Int, major::CompactMajor)
    return ifelse(isone(shape), zero(shape), current)
end
function compact_major(shape::IntTuple, current::Int, major::CompactColMajor)
    return tuple((compact_major(shape[i], current * product(shape[1:(i - 1)]), major) for i in 1:length(shape))...)
end

function compact_major(shape::IntTuple, current::Int, major::CompactRowMajor)
    return tuple((compact_major(shape[i], current * product(shape[(i + 1):(end - 1)]),
                                major) for i in 1:length(shape))...)
end

function compact_major(shape::IntTuple, current::IntTuple, major::CompactMajor)
    length(shape) == length(current) ||
        throw(DimensionMismatch("shape and current must have the same rank"))
    return tuple((compact_major(s, c, major) for (s, c) in zip(shape, current))...)
end

compact_col_major(shape, current=1) = compact_major(shape, current, CompactColMajor())
compact_row_major(shape, current=1) = compact_major(shape, current, CompactRowMajor())

### index_to_coord
function index_to_coord(index::Int, shape::Int, stride::Int)
    @inline
    return ((index - 1) ÷ stride) % shape + 1
end
function index_to_coord(index::Int, shape::Tuple, stride::Tuple)
    length(shape) == length(stride) ||
        throw(DimensionMismatch("shape, and stride must have the same rank"))
    return tuple((index_to_coord(index, s, d) for (s, d) in zip(shape, stride))...)
end
function index_to_coord(index::Int, shape::Tuple, stride::Int)
    return tuple((index_to_coord(index, s, d) for (s, d) in zip(shape,
                                                                compact_col_major(shape,
                                                                                  stride)))...)
end
function index_to_coord(index::Tuple, shape::Tuple, stride::Tuple)
    length(index) == length(shape) == length(stride) ||
        throw(DimensionMismatch("index, shape, and stride must have the same rank"))
    return map(index_to_coord, index, shape, stride)
end

# default stride, compact + column major

function index_to_coord(index::Int, shape::Int)
    @inline
    return index
end
function index_to_coord(index::Int, shape::Tuple)
    return index_to_coord(index, shape, compact_col_major(shape, 1))
end
function index_to_coord(index::Tuple, shape::Tuple)
    length(index) == length(shape) ||
        throw(DimensionMismatch("index and shape must have the same rank"))
    return map(index_to_coord, index, shape)
end

"""
Transoform a coordinate in one shape to a coordinate in another shape.
"""
function coord_to_coord(coord::Tuple, src_shape::Tuple, dst_shape::Tuple)
    length(coord) == length(src_shape) == length(dst_shape) ||
        throw(DimensionMismatch("coord, shape1, and shape2 must have the same rank"))
    return map(coord_to_coord, coord, src_shape, dst_shape)
end
function coord_to_coord(coord, src_shape, dst_shape)
    return index_to_coord(coord_to_index(coord, src_shape), dst_shape)
end

function compact_order(shape::Tuple, order::Tuple, org_shape, org_order)
    return tuple((compact_order(s, o, org_shape, org_order) for (s, o) in zip(shape, order))...)
end
function compact_order(shape, order::Int, org_shape::Tuple, org_order::IntSequence)
    org_order = map(Base.Fix2(-, order), org_order)
    d = productuct(map((s, o) -> ifelse(signbit(o), productuct(s), 1), org_shape,
                       org_order))
    return compact_col_major(shape, d)
end
function compact_order(shape, order)
    iscongruent(shape, order) ||
        throw(DimensionMismatch("shape and order are not congruent"))
    return compact_order(shape, order, flatten(shape), flatten(order))
end
function compact_order(shape, major::CompactMajor)
    return compact_major(shape, 1, major)
end
