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
    q, r = divrem(coord, prod(s))
    return coord_to_index0(r, s, d) + coord_to_index0(q, Base.tail(shape), Base.tail(stride))
end
function coord_to_index0(coord::Tuple, shape::Tuple, stride::Tuple)
    sum(map(coord_to_index0, coord, shape, stride))
end

function coord_to_index(coord, shape, stride)
    coord_to_index0(emap(Base.Fix2(-, 1), coord), shape, stride) + 1
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
    iscongruent(coord, shape) || throw(DimensionMismatch("coord and shape are not congruent"))
    return coord_to_index0_horner(flatten(coord), flatten(shape))
end

function coord_to_index(coord, shape)
    coord_to_index0(emap(Base.Fix2(-, 1), coord), shape) + 1
end

### index_to_coord
function index_to_coord(index::Int, shape::Int, stride::Int)
    @inline
    return ((index - 1) ÷ stride) % shape + 1
end
function index_to_coord(index::Int, shape::Tuple, stride::Tuple)
    length(shape) == length(stride) || throw(DimensionMismatch("shape, and stride must have the same rank"))
    return ((index_to_coord(index, s, d) for (s,d) in zip(shape, stride))...,)
end
function index_to_coord(index::Tuple, shape::Tuple, stride::Tuple)
    length(index) == length(shape) == length(stride) || throw(DimensionMismatch("index, shape, and stride must have the same rank"))
    map(index_to_coord, index, shape, stride)
end
