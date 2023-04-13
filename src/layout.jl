struct Layout{N, Shape, Stride}
    shape::Shape
    stride::Stride

    Layout(shape::IntTuple, stride::IntTuple) = new{rank(shape), typeof(shape), typeof(stride)}(shape, stride)
    Layout(shape::Int, stride::Int) = new{1, typeof(shape), typeof(stride)}(shape, stride)
end

shape(l::Layout) = getfield(l, :shape)
stride(l::Layout) = getfield(l, :stride)

function Base.show(io::IO, l::Layout)
    return print(io, shape(l), ":", stride(l))
end

function (l::Layout)(coord::Tuple)
    if Colon() ∈ coord
        return slice(l, coord)
    end
    return coord_to_index(coord, l.shape, l.stride)
end
function (l::Layout)(index::Int)
    return index_to_coord(index, l.shape, l.stride)
end
function (l::Layout)(c1, c2, c3...)
    return l((c1, c2, c3...))
end


function get_hier_coord(l::Layout, index::Int)
    index_to_coord(index, l.shape, l.stride)
end

function get_congr_coord(l::Layout{N}, index::Int) where N
    coord_to_coord(get_hier_coord(l, index), l.shape, repeat(1, N))
end

function get_linear_coord(l::Layout, index::Int)
    coord_to_index(get_hier_coord(l, index), l.shape)
end

function make_layout(shape::IntTuple, stride::IntTuple)
    Layout(shape, stride)
end
function make_layout(shape::Int, stride::Int)
    Layout(shape, stride)
end
function make_layout(shape::Union{Int, IntTuple})
    Layout(shape, compact_col_major(shape))
end
function make_layout(layouts::Layout...)
    make_layout(shape.(layouts), stride.(layouts))
end
function make_layout(shape, ::CompactColMajor)
    make_layout(shape, compact_col_major(shape))
end
function make_layout(shape, ::CompactRowMajor)
    make_layout(shape, compact_row_major(shape))
end


function make_ordered_layout(shape, order) # The arguments may be static, which is not handled
    make_layout(shape, compact_order(shape, order))
end
function make_ordered_layout(layout::Layout)
    make_ordered_layout(shape(layout), stride(layout))
end

# make_layout_like
# make_fragment_like
# make_identity_layout

function Base.getindex(layout::Layout, Is...)
    return make_layout(getindex(shape(layout), Is...), getindex(stride(layout), Is...))
end

function take(layout::Layout, B::Int, E::Int)
    return make_layout(take(shape(layout), B, E), take(stride(layout), B, E))
end

function flatten(layout::Layout)
    return make_layout(flatten(shape(layout)), flatten(stride(layout)))
end

# avoid overloading Base.size(Int)
function Base.size(layout::Layout{N, Int}) where N
    return shape(layout)
end

function Base.size(layout::Layout)
    return size(shape(layout))
end

function rank(layout::Layout)
    return rank(shape(layout))
end

function depth(layout::Layout)
    return depth(shape(layout))
end

## cosize, negative stride is not supported
function cosize(layout::Layout)
    layout(size(layout))
end

function coord_to_index(coord, layout::Layout)
    return coord_to_index(coord, shape(layout), stride(layout))
end

function slice(coord, layout::Layout)
    return make_layout(slice(coord, shape(layout)), slice(coord, stride(layout)))
end

function slice_and_offset(coord, layout::Layout)
    return slice(coord, layout), coord_to_index(coord, layout)
end

function dice(coord, layout::Layout)
    return make_layout(dice(coord, shape(layout)), dice(coord, stride(layout)))
end

function filter_zeros(layout::Layout)
    return make_layout(filter_zeros(stride(layout), shape(layout)), stride(layout))
end

function append(layout::Layout, x::Layout, N::Int)
    return make_layout(append(shape(layout), shape(x), N), append(stride(layout), stride(x), N))
end

function prepend(layout::Layout, x::Layout, N::Int)
    return make_layout(prepend(shape(layout), shape(x), N), prepend(stride(layout), stride(x), N))
end

function replace(layout::Layout, x::Layout, N::Int)
    return make_layout(replace(shape(layout), shape(x), N), replace(stride(layout), stride(x), N))
end

function group(layout::Layout, B::Int, E::Int)
    return make_layout(group(shape(layout), B, E), group(stride(layout), B, E))
end

# transform_layout
function transform_layout(f::Function, t1::Tuple, t2::Tuple)
    R1 = rank(t1)
    R2 = rank(t2)
    R = (R1 < R2) ? R1 : R2
    make_layout(map(f, t1[1:R], t2[1:R])..., t1[R+1:end]..., t2[R+1:end]...)
end


#coalesce
function bw_coalesce(I::Int, old_shape, old_stride, new_shape, new_stride)
    if I == 0
        if new_shape == 1
            return Layout(1, 0)
        else
            return Layout(new_shape, new_stride)
        end
    elseif old_shape[I] == 1
        return bw_coalesce(I-1, old_shape, old_stride, new_shape, new_stride)
    elseif new_shape == 1
        return bw_coalesce(I-1, old_shape, old_stride, old_shape[I], old_stride[I])
    elseif old_shape[I] * old_stride[I] == new_stride[1]
        return bw_coalesce(I-1, old_shape, old_stride,
                           replace_front(new_shape, old_shape[I] * new_shape[1]),
                           replace_front(new_stride, old_stride[I]))
    else
        return bw_coalesce(I-1, old_shape, old_stride,
                           prepend(new_shape, old_shape[I]),
                            prepend(new_stride, old_stride[I]))
    end
end

function Base.coalesce(layout::Layout)
    flat_shape = flatten(shape(layout))
    flat_stride = flatten(stride(layout))
    bw_coalesce(rank(flat_shape)-1, flat_shape, flat_stride, last(flat_shape), last(flat_stride))
end


#filter

function composition(lhs::Layout, rhs_shape::IntTuple, rhs_stride::IntTuple)
    return let lhs = lhs
        make_layout(map((s,d) -> composition(lhs, s, d), rhs_shape, rhs_stride)...) # Note: there is a closure
    end                                                                             # we assume rank(lhs) == rank(rhs)
end




function composition(lhs::Layout, rhs::Layout)
    return composition(lhs, shape(rhs), stride(rhs))
end

function composition(lhs::Layout, rhs::IntTuple)
    length(lhs.shape) == length(rhs) || throw(DimensionMismatch("shape of lhs and rhs must have the same rank"))
    #transform_layout(lhs, rhs, #)
end
function composition(lhs::Layout, rhs::Colon)
    return lhs
end
function composition(lhs::Layout, rhs)
    return composition(lhs, make_layout(rhs))
end




# Compose

function compose(l1::Layout, l2::Layout)
    return composition(l1, l2)
end
function compose(l1::Layout, l2::Layout, l3::Layout...)
    return composition(l1, (l2, l3...))
end


function withshape(l::Layout, shape::Union{Int, IntTuple})
    return composition(l, make_layout(shape))
end
function withshape(l::Layout, s1, s2, s3...)
    return composition(l, make_layout((s1, s2, s3...)))
end
