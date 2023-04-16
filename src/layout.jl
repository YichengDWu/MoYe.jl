struct Layout{N, Shape, Stride}
    shape::Shape
    stride::Stride

    # in fact shape and stride should be congruent
    function Layout(shape::IntTuple{N}, stride::IntTuple{N}) where {N}
        return new{rank(shape), typeof(shape), typeof(stride)}(shape, stride)
    end
    function Layout(shape::IntType, stride::IntType)
        return new{1, typeof(shape), typeof(stride)}(shape, stride)
    end
end

shape(l::Layout) = getfield(l, :shape)
Base.stride(l::Layout) = getfield(l, :stride)

function Base.show(io::IO, l::Layout)
    return print(io, shape(l), ":", stride(l))
end

# map a logical coordinate to a linear index
function (l::Layout)(coord::IntType)
    return coord_to_index(coord, shape(l), stride(l))
end
function (l::Layout)(@nospecialize coord::IntTuple)
    return coord_to_index(coord, shape(l), stride(l))
end
function (l::Layout)(coord) # coord is fixed with colon
    @assert Colon() ∈ coord
    return slice(l, coord)
end
function (l::Layout)(c1, c2, c3...)
    return l((c1, c2, c3...))
end

# map 1D index to a hier coordinate
function get_hier_coord(l::Layout, index::IntType)
    return index_to_coord(index, l.shape, l.stride)
end

function get_congr_coord(l::Layout{N}, index::IntType) where {N}
    return coord_to_coord(get_hier_coord(l, index), l.shape, repeat(1, N))
end

function get_linear_coord(l::Layout, index::IntType)
    return coord_to_index(get_hier_coord(l, index), l.shape)
end

function make_layout(@nospecialize(shape::IntTuple), @nospecialize(stride::IntTuple))
    return Layout(shape, stride)
end
function make_layout(shape::IntType, stride::IntType)
    return Layout(shape, stride)
end
function make_layout(shape::Union{Int, IntTuple})
    return Layout(shape, compact_col_major(shape))
end
function make_layout(layouts::Layout...)
    return make_layout(shape.(layouts), stride.(layouts)) # concatenation
end
function make_layout(shape, ::CompactColMajor)
    return make_layout(shape, compact_col_major(shape))
end
function make_layout(shape, ::CompactRowMajor)
    return make_layout(shape, compact_row_major(shape))
end

function make_ordered_layout(shape, order) # The arguments may be static, which is not handled
    return make_layout(shape, compact_order(shape, order))
end
function make_ordered_layout(layout::Layout)
    return make_ordered_layout(shape(layout), stride(layout))
end

# make_layout_like
# make_fragment_like
# make_identity_layout

function Base.getindex(layout::Layout, Is::IntType...)
    @inline
    return make_layout(getindex(shape(layout), Is...), getindex(stride(layout), Is...))
end
function Base.getindex(@nospecialize(t::Layout), r::AbstractUnitRange)
    @inline
    return ntuple(i -> t[i + first(r) - 1], length(r))
end

# Layout as iterator
function Base.firstindex(l::Layout)
    return 1
end
function Base.lastindex(l::Layout)
    return rank(l)
end

function Base.first(l::Layout)
    return l[1]
end

function Base.last(l::Layout{N}) where {N}
    return l[N]
end

function Base.length(l::Layout{N}) where {N}
    return N
end

function Base.iterate(l::Layout)
    return l[1], 1
end
function Base.iterate(x::Layout{N}, state) where {N}
    if state == N
        return nothing
    end
    new_state = state + 1
    return (x[new_state], new_state)
end

function take(layout::Layout, B::IntType, E::IntType)
    return make_layout(take(shape(layout), B, E), take(stride(layout), B, E))
end

function flatten(layout::Layout)
    return make_layout(flatten(shape(layout)), flatten(stride(layout)))
end

function Base.size(layout::Layout)
    return capacity(shape(layout))
end
function Base.size(layout::Layout, i::IntType)
    return capacity(shape(layout)[i])
end

function rank(layout::Layout)
    return rank(shape(layout))
end
function rank(layout::Layout, i::IntType)
    return rank(shape(layout)[i])
end

function depth(layout::Layout)
    return depth(shape(layout))
end
function depth(layout::Layout, i::IntType)
    return depth(shape(layout)[i])
end

## cosize, negative stride is not supported
function cosize(layout::Layout)
    return layout(size(layout))
end

function coord_to_index(coord, layout::Layout)
    return coord_to_index(coord, shape(layout), stride(layout))
end

function slice(layout::Layout, coord)
    return make_layout(slice(shape(layout), coord), slice(stride(layout), coord))
end

function slice_and_offset(layout::Layout, coord)
    return slice(layout, coord), coord_to_index(layout, coord)
end

function dice(layout::Layout, coord)
    return make_layout(dice(shape(layout), coord), dice(stride(layout), coord))
end

function append(layout::Layout, x::Layout, N::IntType)
    return make_layout(append(shape(layout), shape(x), N),
                       append(stride(layout), stride(x), N))
end

function append(layout::Layout, N::IntType)
    return append(layout, make_layout(1, 0), N)
end

function prepend(layout::Layout, x::Layout, N::IntType)
    return make_layout(prepend(shape(layout), shape(x), N),
                       prepend(stride(layout), stride(x), N))
end

function prepend(layout::Layout, N::IntType)
    return prepend(layout, make_layout(1, 0), N)
end

function replace(layout::Layout, x::Layout, N::IntType)
    return make_layout(replace(shape(layout), shape(x), N),
                       replace(stride(layout), stride(x), N))
end

function group(layout::Layout, B::IntType, E::IntType)
    return make_layout(group(shape(layout), B, E), group(stride(layout), B, E))
end

# transform_layout
function transform_layout(f::Function, t1, t2)
    R1 = length(t1)
    R2 = length(t2)
    R = (R1 < R2) ? R1 : R2
    return make_layout(map(f, t1[1:R], t2[1:R])..., t1[(R + 1):end]..., t2[(R + 1):end]...)
end

function bw_coalesce(::Val{0}, old_shape, old_stride, new_shape, new_stride)
    new_shape == 1 && return Layout(1, 0)
    return Layout(new_shape, new_stride)
end
function bw_coalesce(::Val{I}, old_shape, old_stride, new_shape, new_stride) where {I}
    if old_shape[I] == 1
        return bw_coalesce(Val(I - 1), old_shape, old_stride, new_shape, new_stride)
    elseif new_shape == 1
        return bw_coalesce(Val(I - 1), old_shape, old_stride, old_shape[I], old_stride[I])
    elseif old_shape[I] * old_stride[I] == new_stride[1]
        return bw_coalesce(Val(I - 1), old_shape, old_stride,
                           replace_front(new_shape, old_shape[I] * new_shape[1]),
                           replace_front(new_stride, old_stride[I]))
    else
        return bw_coalesce(Val(I - 1), old_shape, old_stride,
                           prepend(new_shape, old_shape[I]),
                           prepend(new_stride, old_stride[I]))
    end
end

function Base.coalesce(layout::Layout)
    flat_shape = flatten(shape(layout))
    flat_stride = flatten(stride(layout))
    return bw_coalesce(Val(rank(flat_shape) - 1), flat_shape, flat_stride, last(flat_shape),
                       last(flat_stride))
end
function Base.coalesce(layout::Layout, @nospecialize trg_profile::IntTuple) # respect the target profile
    @assert rank(trg_profile) <= rank(layout)
    return transform_layout(coalesce, layout, trg_profile)
end
function Base.coalesce(layout::Layout, trg_profile)
    return coalesce(layout)
end

function filter_zeros(l::Layout)
    return make_layout(filter_zeros(stride(l), shape(l)), stride(l))
end

function Base.filter(l::Layout)
    return coalesce(filter_zeros(l))
end

# Base case a:b ∘ c:d = c:(b*d)
function composition(lhs::Layout{1, IntType, IntType}, rhs_shape::IntType,
                     rhs_stride::IntType)
    return Layout(rhs_shape, rhs_stride * stride(lhs))
end

# distributivity with concatenation
function composition(lhs::Layout, rhs_shape::IntTuple{N}, rhs_stride::IntTuple{N}) where {N}
    return let lhs = lhs
        make_layout(map((s, d) -> composition(lhs, s, d), rhs_shape, rhs_stride)...) # Note: there is a closure
    end                                                                             # we assume rank(lhs) == rank(rhs)
end

function composition(lhs::Layout, rhs_shape::IntType, rhs_stride::IntType)
    flat_shape = flatten(shape(lhs))
    flat_stride = flatten(stride(lhs))
    if iszero(rhs_stride)
        return Layout(rhs_shape, rhs_stride)
    elseif flat_shape isa Integer
        result_stride = flat_stride * rhs_stride
        return Layout(rhs_shape, result_stride)
    elseif isone(rhs_shape)
        result_shape_0 = flat_shape[1:(end - 1)]
        result_shape_1, rest_shape = foldl((init, si) -> (append(init[1],
                                                                 min(abs(si), init[2])),
                                                          shape_div(init[2], abs(si))),
                                           result_shape_0; init=((), rhs_shape))
        return bw_coalesce(Val(rank(flat_shape) - 1), result_shape_1, flat_stride,
                           rest_shape, last(flat_stride))
    else
        result_shape_0 = flat_shape[1:(end - 1)]
        result_stride_0 = flat_stride[1:(end - 1)]

        result_shape_1, rest_stride = foldl((init, di) -> (append(init[1],
                                                                  shape_div(di, init[2])),
                                                           shape_div(init[2], di)),
                                            result_shape_0; init=((), rhs_stride))

        result_stride_1 = elem_scale(result_stride_0,
                                     shape_div(result_shape_0, result_shape_1))

        result_shape_2, rest_shape = foldl((init, si) -> (append(init[1],
                                                                 min(abs(si), init[2])),
                                                          shape_div(init[2], abs(si))),
                                           result_shape_1; init=((), rhs_shape))
        return bw_coalesce(Val(rank(flat_shape) - 1), result_shape_2, result_stride_1,
                           rest_shape, rest_stride * last(flat_stride))
    end
end

function composition(lhs::Layout, rhs::Layout)
    return composition(lhs, shape(rhs), stride(rhs))
end
function composition(lhs::Layout, @nospecialize rhs::Tuple{Vararg{Layout}})
    @assert rank(rhs) <= length(lhs)
    return make_layout(Iterators.map(composition, lhs, rhs)...)
end
function composition(lhs::Layout, rhs::Colon)
    return lhs
end
function composition(lhs::Layout, rhs)
    return composition(lhs, make_layout(rhs))
end

function compose(l1::Layout, l2::Layout)
    return composition(l1, l2)
end
function compose(l1::Layout, l2::Layout, l3::Layout...)
    return composition(l1, (l2, l3...))
end

function Base.:(∘)(l1::Layout, l2::Layout)
    return compose(l1, l2)
end
function Base.:(∘)(l1::Layout, l2::Layout, l3::Layout...)
    return compose(l1, l2, l3...)
end

function withshape(l::Layout, shape::Union{Int, StaticInt, IntTuple})
    return composition(l, make_layout(shape))
end
function withshape(l::Layout, s1, s2, s3...)
    return composition(l, make_layout((s1, s2, s3...)))
end

function complement(l::Layout, cosize_hi::IntType)
    flat_layout = filter(l)

    if stride(flat_layout) == 0
        return make_layout(cosize_hi)
    end

    R = rank(flat_layout)
    result = (shape(flat_layout), stride(flat_layout), (), (1,))

    if !isone(R)
        function f(init, i)
            curr_stride = minimum(init[2])
            curr_idx = findfirst(==(curr_stride), init[2])
            if isnothing(curr_idx)
                curr_idx = length(init[2])
            end
            curr_shape = init[1][curr_idx]

            return (remove(init[1], curr_idx), remove(init[2], curr_idx),
                    append(init[3], curr_stride ÷ init[4][i]),
                    append(init[4], curr_shape * curr_stride))
        end

        result = foldl(f, ntuple(identity, R - 1); init=result)
    end
    result_stride = last(result)
    result_shape = append(result[3], result[2][1] ÷ back(result_stride))

    rest_stride = result[1][1] * result[2][1]
    return bw_coalesce(Val(R), result_shape, result_stride, cld(cosize_hi, rest_stride),
                       rest_stride)
end

function complement(l::Layout)
    return complement(l, cosize(l))
end

# inverse_seq

# right_inverse
# left_inverse
# max_common_vector

# this is equivalent to make_layout(map(make_layout, l1, l2)...)
function _transpose(layoutA::Layout, layoutB::Layout)
    return make_layout(_transpose(shape(layoutA), shape(layoutB)),
                       _transpose(stride(layoutA), stride(layoutB)))
end

function tile_unzip(layout::Layout, @nospecialize(tile::Tuple))
    return make_layout(zip2_by(shape(layout), tile), zip2_by(stride(layout), tile))
end

function logical_product(layout::Layout, tile::Layout)
    return make_layout(layout,
                       composition(complement(layout, size(layout) * cosize(layout)), tile))
end
function logical_product(layout::Layout, tile::Colon)
    return layout
end
function logical_product(layout::Layout, tile::IntType)
    return logical_product(layout, make_layout(tile))
end
function logical_product(layout::Layout, @nospecialize(tile::IntTuple))
    return transform_layout(logical_product, layout, tile)
end

# zipped_product

# tiled_product
function blocked_product(block::Layout{N}, layout::Layout{M}) where {N, M}
    R = max(N, M)
    padded_block = append(block, R)
    padded_layout = append(layout, R)
    result = logical_product(padded_block, padded_layout)
    return coalesce(_transpose(result[1], result[2]), repeat(1, R))
end

function raked_product(block::Layout{N}, layout::Layout{M}) where {N, M}
    R = max(N, M)
    padded_block = append(block, R)
    padded_layout = append(layout, R)
    result = logical_product(padded_block, padded_layout)
    return coalesce(_transpose(result[2], result[1]), repeat(static(1), R))
end

# tile_to_shape

# upcast

# downcast

# recast

function logical_divide(layout::Layout, tile::Layout)
    return composition(layout, make_layout(tile, complement(tile, size(layout))))
end
function logical_divide(layout::Layout, @nospecialize tile::Tuple)
    length(tile) <= rank(layout) || throw(DimensionMismatch("too many modes in tile"))
    return transform_layout(logical_divide, layout, tile)
end
function logical_divide(layout::Layout, tile::Colon)
    return layout
end
function logical_divide(layout::Layout, tile::IntType)
    return logical_divide(layout, make_layout(tile))
end

function zipped_divide(layout::Layout, tile)
    return tile_unzip(logical_divide(layout, tile), tile)
end

function tiled_divide(layout::Layout, tile)
    d = zipped_divide(layout, tile)
    R = rank(d, 2)
    return d(:, repeat(:, R))
end
