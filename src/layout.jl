struct Layout{N, Shape, Stride}
    shape::Shape
    stride::Stride

    # shape and stride must be congruent
    function Layout(shape::IntTuple{N}, stride::IntTuple{N}) where {N}
        return new{rank(shape), typeof(shape), typeof(stride)}(shape, stride)
    end
    function Layout(shape::IntType, stride::IntType)
        return new{1, typeof(shape), typeof(stride)}(shape, stride)
    end
end

"""
A tuple of `Layout` or `Colon`.
"""
const Tile{N} = Tuple{Vararg{Union{Colon, Layout}, N}}
const GenIntTuple = Union{Int, StaticInt, IntTuple}
const StaticLayout{N} = Layout{N, <:StaticIntTuple{N}, <:StaticIntTuple{N}}

shape(l::Layout) = getfield(l, :shape)
Base.stride(l::Layout) = getfield(l, :stride)

function Base.show(io::IO, l::Layout)
    return print(io, shape(l), ":", stride(l))
end

Static.is_static(l::Layout) = is_static(shape(l)) && is_static(stride(l))

# map a logical coordinate to a linear index
function (l::Layout)(@nospecialize coord::IntType)
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
function get_hier_coord(l::Layout, @nospecialize index::IntType)
    return index_to_coord(index, l.shape, l.stride)
end

function get_congr_coord(l::Layout{N}, @nospecialize index::IntType) where {N}
    return coord_to_coord(get_hier_coord(l, index), l.shape, repeat(1, N))
end

function get_linear_coord(l::Layout, @nospecialize index::IntType)
    return coord_to_index(get_hier_coord(l, index), l.shape)
end

function make_layout(@nospecialize(shape::IntTuple), @nospecialize(stride::IntTuple))
    return Layout(shape, stride)
end
function make_layout(shape::IntType, stride::IntType)
    return Layout(shape, stride)
end
function make_layout(shape::GenIntTuple)
    return Layout(shape, compact_col_major(shape))
end
function make_layout(layouts::Layout...)
    return make_layout(shape.(layouts), stride.(layouts)) # concatenation
end
function make_layout(shape::GenIntTuple, ::Type{GenColMajor})
    return make_layout(shape, compact_col_major(shape))
end
function make_layout(shape::GenIntTuple, ::Type{GenRowMajor})
    return make_layout(shape, compact_row_major(shape))
end

"""
    Layout(shape, stride=nothing)

Construct a static layout with the given shape and stride.

## Arguments

  - `shape`: a tuple of integers or a single integer
  - `stride`: a tuple of integers, a single integer, `GenColMajor` or `GenRowMajor`
"""
macro Layout(expr1, expr2=nothing)
    if expr2 === nothing
        layout_call = :(make_layout(static($expr1)))
    elseif expr2 isa Symbol
        layout_call = :(make_layout(static($expr1), $expr2))
    else
        expr2
        layout_call = :(make_layout(static($expr1), static($expr2)))
    end
    return layout_call
end

"""
    make_ordered_layout(shape, order)
    make_ordered_layout(layout)

Construct a compact layout with the given shape and the stride is following the given order.

## Examples

```julia
julia> CuTe.make_ordered_layout((3, 5), (2, 6))
(3, 5):(static(1), 3)

julia> CuTe.make_ordered_layout((3, 5), (10, 2))
(3, 5):(5, static(1))
```
"""
function make_ordered_layout(shape, order) # The arguments may be static, which is not handled
    return make_layout(shape, compact_order(shape, order))
end
function make_ordered_layout(layout::Layout)
    return make_ordered_layout(shape(layout), stride(layout))
end

# make_layout_like

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

function flatten(layout::Layout)
    return make_layout(flatten(shape(layout)), flatten(stride(layout)))
end

function Base.size(layout::Layout)
    return capacity(shape(layout))
end
function Base.size(layout::Layout, i::Int)
    return capacity(shape(layout)[i])
end

function rank(layout::Layout)
    return rank(shape(layout))
end
function rank(layout::Layout, i::Int)
    return rank(shape(layout)[i])
end

function depth(layout::Layout)
    return depth(shape(layout))
end
function depth(layout::Layout, i::Int)
    return depth(shape(layout)[i])
end

## cosize, negative stride is not supported
function cosize(layout::Layout)
    return layout(size(layout))
end

function coord_to_index(layout::Layout, coord)
    return coord_to_index(coord, shape(layout), stride(layout))
end

function coord_to_index0(layout::Layout, coord)
    return coord_to_index0(coord, shape(layout), stride(layout))
end

function slice(layout::Layout, coord)
    return make_layout(slice(shape(layout), coord), slice(stride(layout), coord))
end

function slice_and_offset(layout::Layout, coord)
    return slice(layout, coord), coord_to_index(layout, coord) - static(1)
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

function transform_layout(f::Function, t1, t2)
    R1 = length(t1)
    R2 = length(t2)
    R = (R1 < R2) ? R1 : R2
    return make_layout(map(f, t1[1:R], t2[1:R])..., t1[(R + 1):end]..., t2[(R + 1):end]...)
end

function bw_coalesce(::Val{0}, old_shape, old_stride, new_shape, new_stride)
    new_shape == 1 && return Layout(one(new_shape), zero(new_shape))
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
    elseif flat_shape isa IntType
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

function withshape(l::Layout, shape::GenIntTuple)
    return composition(l, make_layout(shape))
end
function withshape(l::Layout, s1, s2, s3...)
    return composition(l, make_layout((s1, s2, s3...)))
end

function _complement(shape::IntType, stride::IntType, cosize_hi::IntType)
    stride == 0 && return make_layout(cosize_hi)
    rest_stride = shape * stride
    return bw_coalesce(Val(1), (stride,), (one(shape),), cld(cosize_hi, rest_stride),
                       rest_stride)
end
function _complement(shape::IntTuple{R}, stride::IntTuple{R}, cosize_hi::IntType) where {R}
    if stride == 0
        return make_layout(cosize_hi)
    end

    curr_stride, curr_idx = findmin(stride)
    curr_shape = shape[curr_idx]

    result = (remove(shape, curr_idx), remove(stride, curr_idx), tuple(curr_stride),
              tuple(one(curr_stride), curr_shape * curr_stride))

    function f(init, i)
        curr_stride, curr_idx = findmin(init[2])
        curr_shape = init[1][curr_idx]

        return (remove(init[1], curr_idx), remove(init[2], curr_idx),
                append(init[3], curr_stride ÷ init[4][i]),
                append(init[4], curr_shape * curr_stride))
    end

    result = foldl(f, ntuple(x -> x + 1, R - 2); init=result)
    result_stride = last(result)
    result_shape = append(result[3], result[2][1] ÷ back(result_stride))

    rest_stride = result[1][1] * result[2][1]
    return bw_coalesce(Val(R), result_shape, result_stride, cld(cosize_hi, rest_stride),
                       rest_stride)
end

function complement(l::Layout, cosize_hi::IntType)
    filter_layout = filter(l)
    return _complement(shape(filter_layout), stride(filter_layout), cosize_hi)
end

function complement(l::Layout)
    filter_layout = filter(l)
    return _complement(shape(filter_layout), stride(filter_layout), cosize(filter_layout))
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

"""
    logical_product(A::Layout, B::Layout)

Compute the logical product of two layouts. Indexing through the first mode of the new layout
corresponds to indexing through `A` and indexing through the second mode corresponds to indexing
through `B`.

```julia
julia> print_layout(tile)
tile = make_layout((2,2), (1,2));

julia> print_layout(matrix_of_tiles)
(2, 2):(1, 2)
      1   2
    +---+---+
 1  | 1 | 3 |
    +---+---+
 2  | 2 | 4 |
    +---+---+

julia> print_layout(logical_product(tile, matrix_of_tiles));
matrix_of_tiles = make_layout((3,4), (4,1));
```
"""
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
function logical_product(layout::Layout, @nospecialize(tile::Tuple))
    return transform_layout(logical_product, layout, tile)
end

function zipped_product(layout::Layout, tile::Tile)
    return tile_unzip(logical_product(layout, tile), tile)
end

function tiled_product(layout::Layout, tile::Tile{N}) where {N}
    d = zipped_product(layout, tile)
    return d(:, repeat(:, N))
end

"""
    blocked_product(tile::Layout, matrix_of_tiles::Layout, coalesce_result::Bool=false)

Compute the blocked product of two layouts. Indexing through the first mode of the new layout
corresponds to indexing through the cartesian product of the first mode of `tile` and the first
mode of `matrix_of_tiles`. Indexing through the second mode is similar. If `coalesce_result` is
true, then the result is coalesced.

```julia
julia> tile = make_layout((2, 2), (1, 2));

julia> matrix_of_tiles = make_layout((3, 4), (4, 1));

julia> print_layout(blocked_product(tile, matrix_of_tiles))

```
"""
function blocked_product(block::Layout{N}, layout::Layout{M},
                         coalesce_result::Bool=false) where {N, M}
    R = max(N, M)
    padded_block = append(block, R)
    padded_layout = append(layout, R)
    result = logical_product(padded_block, padded_layout)
    result = _transpose(result[1], result[2])
    coalesce_result && return coalesce(result, repeat(static(1), R))
    return result
end

"""
    raked_product(tile::Layout, matrix_of_tiles::Layout, coalesce_result::Bool=false)

The tile is shattered or interleaved with the matrix of tiles.

```julia
julia> tile = make_layout((2, 2), (1, 2));

julia> matrix_of_tiles = make_layout((3, 4), (4, 1));

julia> print_layout(raked_product(tile, matrix_of_tiles))
((3, 2), (4, 2)):((16, 1), (4, 2))
       1    2    3    4    5    6    7    8
    +----+----+----+----+----+----+----+----+
 1  |  1 |  5 |  9 | 13 |  3 |  7 | 11 | 15 |
    +----+----+----+----+----+----+----+----+
 2  | 17 | 21 | 25 | 29 | 19 | 23 | 27 | 31 |
    +----+----+----+----+----+----+----+----+
 3  | 33 | 37 | 41 | 45 | 35 | 39 | 43 | 47 |
    +----+----+----+----+----+----+----+----+
 4  |  2 |  6 | 10 | 14 |  4 |  8 | 12 | 16 |
    +----+----+----+----+----+----+----+----+
 5  | 18 | 22 | 26 | 30 | 20 | 24 | 28 | 32 |
    +----+----+----+----+----+----+----+----+
 6  | 34 | 38 | 42 | 46 | 36 | 40 | 44 | 48 |
    +----+----+----+----+----+----+----+----+
```
"""
function raked_product(block::Layout{N}, layout::Layout{M},
                       coalesce_result::Bool=false) where {N, M}
    R = max(N, M)
    padded_block = append(block, R)
    padded_layout = append(layout, R)
    result = logical_product(padded_block, padded_layout)
    result = _transpose(result[2], result[1])
    coalesce_result && return coalesce(result, repeat(static(1), R))
    return result
end

# tile_to_shape

# upcast

# downcast

# recast

"""
    logical_divide(layout::Layout, tile::Tile)

Gather the elements of `layout` along all modes into blocks according to `tile`.

```julia
julia> raked_prod = make_layout(((3, 2), (4, 2)), ((16, 1), (4, 2)));

julia> print_layout(raked_prod)
((3, 2), (4, 2)):((16, 1), (4, 2))
       1    2    3    4    5    6    7    8
    +----+----+----+----+----+----+----+----+
 1  |  1 |  5 |  9 | 13 |  3 |  7 | 11 | 15 |
    +----+----+----+----+----+----+----+----+
 2  | 17 | 21 | 25 | 29 | 19 | 23 | 27 | 31 |
    +----+----+----+----+----+----+----+----+
 3  | 33 | 37 | 41 | 45 | 35 | 39 | 43 | 47 |
    +----+----+----+----+----+----+----+----+
 4  |  2 |  6 | 10 | 14 |  4 |  8 | 12 | 16 |
    +----+----+----+----+----+----+----+----+
 5  | 18 | 22 | 26 | 30 | 20 | 24 | 28 | 32 |
    +----+----+----+----+----+----+----+----+
 6  | 34 | 38 | 42 | 46 | 36 | 40 | 44 | 48 |
    +----+----+----+----+----+----+----+----+

julia> subtile = (Layout(2, 3), Layout(2, 4)); # gather 2 elements with stride 3 along the first mode
       # and 2 elements with stride 4 along the second mode

julia> print_layout(logical_divide(raked_prod, subtile))
((2, 3), (2, 4)):((1, 16), (2, 4))
       1    2    3    4    5    6    7    8
    +----+----+----+----+----+----+----+----+
 1  |  1 |  3 |  5 |  7 |  9 | 11 | 13 | 15 |
    +----+----+----+----+----+----+----+----+
 2  |  2 |  4 |  6 |  8 | 10 | 12 | 14 | 16 |
    +----+----+----+----+----+----+----+----+
 3  | 17 | 19 | 21 | 23 | 25 | 27 | 29 | 31 |
    +----+----+----+----+----+----+----+----+
 4  | 18 | 20 | 22 | 24 | 26 | 28 | 30 | 32 |
    +----+----+----+----+----+----+----+----+
 5  | 33 | 35 | 37 | 39 | 41 | 43 | 45 | 47 |
    +----+----+----+----+----+----+----+----+
 6  | 34 | 36 | 38 | 40 | 42 | 44 | 46 | 48 |
    +----+----+----+----+----+----+----+----+
```
"""
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

"""
    zipped_divide(layout::Layout, tile::Tile)

Compute the logical division of `layout` by `tile`, then flatten the blocks into single
mode and the rest into another mode.

```julia
julia> raked_prod = make_layout(((3, 2), (4, 2)), ((16, 1), (4, 2)));

julia> print_layout(raked_prod)

julia> subtile = (Layout(2, 3), Layout(2, 4)); # gather 2 elements with stride 3 along the first mode
       # and 2 elements with stride 4 along the second mode
((3, 2), (4, 2)):((16, 1), (4, 2))
       1    2    3    4    5    6    7    8
    +----+----+----+----+----+----+----+----+
 1  |  1 |  5 |  9 | 13 |  3 |  7 | 11 | 15 |
    +----+----+----+----+----+----+----+----+
 2  | 17 | 21 | 25 | 29 | 19 | 23 | 27 | 31 |
    +----+----+----+----+----+----+----+----+
 3  | 33 | 37 | 41 | 45 | 35 | 39 | 43 | 47 |
    +----+----+----+----+----+----+----+----+
 4  |  2 |  6 | 10 | 14 |  4 |  8 | 12 | 16 |
    +----+----+----+----+----+----+----+----+
 5  | 18 | 22 | 26 | 30 | 20 | 24 | 28 | 32 |
    +----+----+----+----+----+----+----+----+
 6  | 34 | 38 | 42 | 46 | 36 | 40 | 44 | 48 |
    +----+----+----+----+----+----+----+----+

julia> print_layout(zipped_divide(raked_prod, subtile))

```
"""
function zipped_divide(layout::Layout, tile::Tile)
    return tile_unzip(logical_divide(layout, tile), tile)
end

"""
    tiled_divide(layout::Layout, tile::Tile)

Similar to `zipped_divide`, but upack the second mode into multiple modes.
"""
function tiled_divide(layout::Layout, tile::Tile)
    d = zipped_divide(layout, tile)
    R = rank(d, 2)
    return d(:, repeat(:, R))
end

function tile(l1::Layout, l2::Layout)
    return tiled_divide(l1, l2)
end
function tile(l1::Layout, l2::Layout, l3::Layout...)
    return tiled_divide(l1, (l2, l3...))
end

"""
    make_fragment_like(::Layout)

Make a compact layout of the same shape with the first mode being col-major, and with the rest
following the given order.
"""
make_fragment_like(layout::StaticLayout{1}) = make_layout(shape(layout))
function make_fragment_like(layout::StaticLayout{R}) where {R}
    return tiled_product(make_layout(shape(layout)[1]),
                         tuple(make_ordered_layout(make_layout(layout[2:end]...))...))
end
make_fragment_like(layout::Layout) = make_layout(shape(layout))
make_fragment_like(shape::GenIntTuple) = make_layout(shape)
