function coord_to_index0(coord::Int, shape::Int, stride::Int)
    @inline
    coord  * stride
end
function coord_to_index0(coord::Int, shape::Tuple, stride::Tuple)
    s, d = first(shape), first(stride)
    if length(shape) == length(stride) == 1
        return coord_to_index0(coord, s, d)
    end
    q, r = divrem(coord, prod(s))
    return coord_to_index0(r, s, d) + coord_to_index0(q, Base.tail(shape), Base.tail(stride))
end

function coord_to_index0(coord::Tuple, shape::Tuple, stride::Tuple)
    mapreduce(coord_to_index0, +, coord, shape, stride)
end

function coord_to_index(coord, shape, stride)
    coord_to_index0(emap(Base.Fix2(-, 1), coord), shape, stride) + 1
end
