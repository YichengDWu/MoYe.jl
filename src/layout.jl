"""
    Layout{N, Shape, Stride}

A `Layout` is a pair of `Shape` and `Stride` tuples.  The `Shape` tuple
contains the number of elements in each dimension, and the `Stride` tuple
contains the number of elements to skip to get to the next element in each
dimension.

## Fields
  - `shape`.
  - `stride`.

## Indexing

A `Layout` can be indexed with three types of indices:

  - `Int`: a linear index in a column-major order.
  - `IntTuple`: a hierarchical index. It has the exact hierarchical structure as defined by the `Shape`.
  - `IntTuple`: a congruent index. A tuple of `N` mixes hierarchical and linear indices along each dimension.

## Examples
```julia
julia> layout = Layout((4, (2, 2)), (2, (1, 8)));

julia> print_layout(ans)
(4, (2, 2)):(2, (1, 8))
       1    2    3    4
    +----+----+----+----+
 1  |  1 |  2 |  9 | 10 |
    +----+----+----+----+
 2  |  3 |  4 | 11 | 12 |
    +----+----+----+----+
 3  |  5 |  6 | 13 | 14 |
    +----+----+----+----+
 4  |  7 |  8 | 15 | 16 |
    +----+----+----+----+

julia> layout(6) # linear index
4

julia> layout((2,2)) # hierarchical index
4

julia> layout((2,(2,1))) # congruent index
4
```
"""
struct Layout{N, Shape, Stride}
    shape::Shape
    stride::Stride

    # shape and stride must be congruent
    function Layout(shape::IntTuple{N}, stride::IntTuple{N}) where {N}
        return new{N, typeof(shape), typeof(stride)}(shape, stride)
    end
    function Layout(shape::IntType, stride::IntType)
        return new{1, typeof(shape), typeof(stride)}(shape, stride)
    end
end

"""
A tuple of `Layout`s, `Colon`s, integers, or tiles.
"""
const Tile{N} = Tuple{
                      Vararg{
                             Union{Colon, Layout, Int, StaticInt,
                                   Tuple{Vararg{Union{Colon, Layout, Int, StaticInt}}}}, N}}
const StaticLayout{N, S, R} = Layout{N, S, R
                                     } where {S <: Union{StaticInt, StaticIntTuple{N}},
                                              R <: Union{StaticInt, StaticIntTuple{N}}}

@inline function StaticLayout{N, S, R}() where {N, S <: StaticIntTuple{N},
                                                R <: StaticIntTuple{N}}
    return Layout(make_tuple(S), make_tuple(R))
end

shape(l::Layout) = getfield(l, :shape)
shape(l, i::IntType) = getindex(shape(l), i)
shape(l::Type{<:StaticLayout{N, S, R}}) where {N, S, R} = S
Base.stride(l::Layout) = getfield(l, :stride)
Base.stride(l::Layout, i::IntType) = getindex(stride(l), i)
Base.stride(l::Type{<:StaticLayout{N, S, R}}) where {N, S, R} = R
Static.static(l::Layout) = make_layout(static(shape(l)), static(stride(l)))

Static.is_static(l::Layout) = dynamic(is_static(shape(l))) && dynamic(is_static(stride(l)))

# map a logical coordinate to a linear index
function (l::Layout)(@nospecialize coord::IntType)
    return coord_to_index(coord, shape(l), stride(l))
end
function (l::Layout)(@nospecialize coord::IntTuple)
    return coord_to_index(coord, shape(l), stride(l))
end
@generated function (l::Layout)(coord) # coord is fixed with colon
    @assert hascolon(coord)
    return :(slice(l, coord))
end
function (l::Layout)(c1, c2, c3...)
    return l((c1, c2, c3...))
end

# map 1D index to a hier coordinate
function get_hier_coord(l::Layout, @nospecialize index::Union{Integer, StaticInt})
    @inline
    return index_to_coord(index, l.shape, l.stride)
end

"""
    get_congr_coord(l::Layout, index::Integer)

Get the flat congruent coordinate from the physical index `index`.
"""
function get_congr_coord(l::Layout{N},
                         @nospecialize index::Union{Integer, StaticInt}) where {N}
    @inline
    return coord_to_coord(get_hier_coord(l, index), l.shape, ntuple(Returns(One()), Val(N)))
end

function get_linear_coord(l::Layout, @nospecialize index::Union{Integer, StaticInt})
    @inline
    return coord_to_index(get_hier_coord(l, index), l.shape)
end

"""
    make_layout(shape::IntTuple, stride::IntTuple)
    make_layout(shape::IntTuple, major=GenColMajor)

Construct a layout with the given shape and stride. If the stride is not given, it is set to
col-major compact stride. See alse [`GenColMajor`](@ref) and [`GenRowMajor`](@ref).
"""
function make_layout(shape::IntTuple, stride::IntTuple)
    @inline
    return Layout(shape, stride)
end
function make_layout(shape::IntType, stride::IntType)
    @inline
    return Layout(shape, stride)
end
function make_layout(shape::GenIntTuple)
    @inline
    return Layout(shape, compact_col_major(shape))
end

"""
    make_layout(::Layouts...)

Concatenate layouts into a single layout.
"""
function make_layout(layouts::Layout...)
    return make_layout(map(shape, layouts), map(stride, layouts)) # concatenation
end
function make_layout(shape::GenIntTuple, ::Type{GenColMajor})
    return make_layout(shape, compact_col_major(shape))
end
function make_layout(shape::GenIntTuple, ::Type{GenRowMajor})
    return make_layout(shape, compact_row_major(shape))
end
function make_layout(shape::Type{S}, stride::Type{D}) where {N, S <: StaticIntTuple{N},
                                                             D <: StaticIntTuple{N}}
    @inline
    return Layout{N, S, D}
end

"""
    cat(::Layouts...)

Concatenate layouts into a single layout.
"""
function Base.cat(layouts::Layout...)
    return make_layout(map(shape, layouts), map(stride, layouts))
end
"""
    @Layout(shape, stride=nothing)

Construct a static layout with the given shape and stride.

## Arguments

  - `shape`: a tuple of integers or a single integer
  - `stride`: a tuple of integers, a single integer, [`GenColMajor`](@ref) or [`GenRowMajor`](@ref)
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
julia> MoYe.make_ordered_layout((3, 5), (2, 6))
(3, 5):(static(1), 3)

julia> MoYe.make_ordered_layout((3, 5), (10, 2))
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
"""
    getindex(layout::Layout, Is...)

Get the sub-layout of `layout` with the given indices.
"""
function Base.getindex(layout::Layout, Is::IntType...)
    @inline
    return make_layout(getindex(shape(layout), Is...), getindex(stride(layout), Is...))
end
function Base.getindex(@nospecialize(t::Layout), r::AbstractUnitRange)
    @inline
    return ntuple(i -> t[i + first(r) - 1], length(r))
end
function Base.getindex(layout::Type{<:StaticLayout}, ::StaticInt{I}) where {I}
    @inline
    return make_layout(getindex(shape(layout).parameters, I), getindex(stride(layout).parameters, I))
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

"""
    flatten(layout::Layout)

Remove the hierarchy of the layout and make it a flat layout.
## Examples

```julia
julia> layout = make_layout(((4, 3), 1), ((3, 1), 0))
((4, 3), 1):((3, 1), 0)


julia> print(flatten(layout))
(4, 3, 1):(3, 1, 0)
```
"""
function flatten(layout::Layout)
    return make_layout(flatten(shape(layout)), flatten(stride(layout)))
end

"""
    size(::Layout)
    size(::Layout, i::Union{Int, StaticInt})

Get the cardinality of the domain of the layout. See also [`cosize`](@ref).
"""
function Base.size(layout::Layout)
    return capacity(shape(layout))
end
function Base.size(layout::Layout, i::Int)
    return capacity(shape(layout)[i])
end
function Base.size(layout::Type{<:StaticLayout})
    return capacity(shape(layout))
end
function Base.size(layout::Type{<:StaticLayout}, ::StaticInt{I}) where {I}
    return capacity(shape(layout).parameters[i])
end

"""
    rank(::Layout)
    rank(::Layout, i::Union{Int, StaticInt})

Get the rank, i.e., the dimensionality, of the layout.
"""
function rank(layout::Layout)
    return rank(shape(layout))
end
function rank(layout::Layout, i::Int)
    return rank(shape(layout)[i])
end
function rank(::Type{<:Layout{N}}) where {N}
    return N
end

"""
    depth(::Layout)
    depth(::Layout, i::Union{Int, StaticInt})

Get the depth of the hierarchy of the layout. For example, the depth of `(1,2)` is 1, and the depth of `((1,2),3)` is 2.
"""
function depth(layout::Layout)
    return depth(shape(layout))
end
function depth(layout::Layout, i::IntType)
    return depth(shape(layout)[i])
end

"""
    cosize(::Layout)
    cosize(::Layout, i::Union{Int, StaticInt})

Get the cardinality of the codomain of the layout. See also [`size`](@ref).
"""
function cosize(layout::Layout)
    return layout(size(layout)) ## Note: negative stride is not supported
end

function coord_to_index(layout::Layout, coord)
    return coord_to_index(coord, shape(layout), stride(layout))
end

function coord_to_index0(layout::Layout, coord)
    return coord_to_index0(coord, shape(layout), stride(layout))
end

@inline iscompatible(a::Layout, b::Layout) = iscompatible(shape(a), shape(b))

function slice(layout::Layout, coord)
    return make_layout(slice(shape(layout), coord), slice(stride(layout), coord))
end

function slice_and_offset(layout::Layout, coord)
    idx = coord_to_index(layout, coord)
    return slice(layout, coord), (idx - one(idx))
end

function dice(layout::Layout, coord)
    return make_layout(dice(shape(layout), coord), dice(stride(layout), coord))
end

function append(layout::Layout{N}, ::Layout, ::StaticInt{N}) where {N}
    return layout
end
function append(layout::Layout{N}, ::StaticInt{N}) where {N}
    return layout
end
function append(layout::Layout, x::Layout, N::StaticInt)
    return make_layout(append(shape(layout), shape(x), N),
                       append(stride(layout), stride(x), N))
end
function append(layout::Layout, N::StaticInt)
    return append(layout, @Layout(1, 0), N)
end

function prepend(layout::Layout, x::Layout, N::StaticInt)
    return make_layout(prepend(shape(layout), shape(x), N),
                       prepend(stride(layout), stride(x), N))
end
function prepend(layout::Layout, N::StaticInt)
    return prepend(layout, @Layout(1, 0), N)
end

function replace(layout::Layout, x::Layout, N::IntType)
    return make_layout(replace(shape(layout), shape(x), N),
                       replace(stride(layout), stride(x), N))
end

function group(layout::Layout, B::IntType, E::IntType)
    return make_layout(group(shape(layout), B, E), group(stride(layout), B, E))
end

@generated function transform_layout(f, t1, t2)
    R1 = rank(t1)
    R2 = rank(t2)

    expr = Expr(:call, :make_layout)
    for i in 1:min(R1, R2)
        push!(expr.args, Expr(:call, :f, :(t1[$i]), :(t2[$i])))
    end

    if R1 < R2
        for i in (R1 + 1):R2
            push!(expr.args, :(t2[$i]))
        end
    elseif R1 > R2
        for i in (R2 + 1):R1
            push!(expr.args, :(t1[$i]))
        end
    end

    return quote
        Base.@_inline_meta
        $expr
    end
end

@generated function transform_layout(f, t1, t2, ::StaticInt{N}) where {N}
    expr = Expr(:call, :make_layout)
    for i in 1:N
        push!(expr.args, Expr(:call, :f, :(t1[$i]), :(t2[$i])))
    end
    return expr
end

@generated function bw_coalesce(::StaticInt{I}, old_shape::StaticIntTuple,
                                                old_stride::StaticIntTuple,
                                                new_shape::GenStaticIntTuple,
                                                new_stride::GenStaticIntTuple) where {I}
    for i in I:-1:1
        if old_shape.parameters[i] == One
            continue
        elseif new_shape <: StaticInt && new_shape == One
            new_shape = old_shape.parameters[i]
            new_stride = old_stride.parameters[i]
        elseif old_shape.parameters[i] * old_stride.parameters[i] == ifelse(new_stride <: Tuple, new_stride.parameters[1], new_stride)
            new_shape = replace_front(new_shape,
                                      old_shape.parameters[i] * ifelse(new_shape <: Tuple, new_shape.parameters[1], new_shape))
            new_stride = replace_front(new_stride, old_stride.parameters[i])
        else
            new_shape = prepend(new_shape, old_shape.parameters[i])
            new_stride = prepend(new_stride, old_stride.parameters[i])
        end
    end

    if new_shape == One
        return :($(Layout(One(), Zero())))
    else
        return :($(Layout(make_tuple(new_shape), make_tuple(new_stride))))
    end
end

function bw_coalesce(::StaticInt{0}, old_shape::StaticIntTuple, old_stride::StaticIntTuple,
                     new_shape::GenStaticIntTuple, new_stride::GenStaticIntTuple)
    return Layout(new_shape, new_stride)
end
function bw_coalesce(::StaticInt{0}, old_shape::StaticIntTuple, old_stride::StaticIntTuple,
                     new_shape::StaticInt{1}, new_stride::StaticInt)
    return Layout(one(new_shape), zero(new_shape))
end
function bw_coalesce(::StaticInt{0}, old_shape, old_stride, new_shape::StaticInt{1},
                     new_stride)
    return Layout(One(), Zero())
end
function bw_coalesce(::StaticInt{0}, old_shape, old_stride, new_shape, new_stride)
    return Layout(new_shape, new_stride)
end

Base.@assume_effects :total function bw_coalesce(I::StaticInt, old_shape, old_stride,
                                                 new_shape, new_stride)
    if isa(old_shape[I], StaticInt) && old_shape[I] == One()
        return bw_coalesce(I - One(), old_shape, old_stride, new_shape, new_stride)
    elseif isa(new_shape, StaticInt) && new_shape == One()
        return bw_coalesce(I - One(), old_shape, old_stride, old_shape[I], old_stride[I])
    elseif old_shape[I] * old_stride[I] == new_stride[1]
        return bw_coalesce(I - One(), old_shape, old_stride,
                           replace_front(new_shape, old_shape[I] * new_shape[1]),
                           replace_front(new_stride, old_stride[I]))
    else
        return bw_coalesce(I - One(), old_shape, old_stride,
                           prepend(new_shape, old_shape[I]),
                           prepend(new_stride, old_stride[I]))
    end
end

"""
    coalesce(layout::Layout)

Coalesce the layout by merging adjacent dimensions with stride 1.

## Examples

```julia
julia> layout = @Layout (2, (1, 6)) (1, (6, 2))
(static(2), (static(1), static(6))):(static(1), (static(6), static(2)))


julia> print(coalesce(layout))
static(12):static(1)
```
"""
function Base.coalesce(layout::Layout)
    flat_shape = flatten(shape(layout))
    flat_stride = flatten(stride(layout))
    return bw_coalesce(static(rank(flat_shape) - 1), flat_shape, flat_stride,
                       last(flat_shape), last(flat_stride))
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

# shortcuts
function composition(lhs_shape::IntType, lhs_stride::IntType, rhs_shape::IntType,
                     rhs_stride::StaticInt{0})
    return Layout(rhs_shape, rhs_stride)
end
# Base case a:b ∘ c:d = c:(b*d)
function composition(lhs_shape::IntType, lhs_stride::IntType, rhs_shape::IntType,
                     rhs_stride::IntType)
    result_stride = lhs_stride * rhs_stride
    return Layout(rhs_shape, result_stride)
end

function composition(lhs_shape::Tuple, lhs_stride::Tuple, rhs_shape::IntType,
                     rhs_stride::StaticInt{1})
    result_shape_0 = Base.front(lhs_shape)
    result_shape_1, rest_shape = _foldl((init, si) -> (append(init[1],
                                                              min(abs(si), init[2])),
                                                       shape_div(init[2], abs(si))),
                                        result_shape_0, ((), rhs_shape))
    return bw_coalesce(static(rank(lhs_shape) - 1), result_shape_1, lhs_stride, rest_shape,
                       last(lhs_stride))
end

function composition(lhs_shape::Tuple, lhs_stride::Tuple, rhs_shape::IntType,
                     rhs_stride::StaticInt{0})
    return Layout(rhs_shape, rhs_stride)
end
function composition(lhs_shape::Tuple, lhs_stride::Tuple, rhs_shape::IntType,
                     rhs_stride::IntType)
    result_shape_0 = Base.front(lhs_shape)
    result_stride_0 = Base.front(lhs_stride)

    result_shape_1, rest_stride = _foldl((init, di) -> (append(init[1],
                                                               shape_div(di, init[2])),
                                                        shape_div(init[2], di)),
                                         result_shape_0, ((), rhs_stride))

    result_stride_1 = elem_scale(result_stride_0, shape_div(result_shape_0, result_shape_1))

    result_shape_2, rest_shape = _foldl((init, si) -> (append(init[1],
                                                              min(abs(si), init[2])),
                                                       shape_div(init[2], abs(si))),
                                        result_shape_1, ((), rhs_shape))
    return bw_coalesce(static(rank(lhs_shape) - 1), result_shape_2, result_stride_1,
                       rest_shape, rest_stride * last(lhs_stride))
end

# distributivity with concatenation
@generated function composition(lhs_shape, lhs_stride, rhs_shape::IntTuple{N},
                                rhs_stride::IntTuple{N}) where {N}
    expr = Expr(:call, :make_layout)
    for i in 1:N
        push!(expr.args,
              :(MoYe.composition(lhs_shape, lhs_stride, rhs_shape[$i], rhs_stride[$i])))
    end
    return expr
end

function composition(lhs::Layout, rhs::Layout)
    flat_shape = flatten(shape(lhs))
    flat_stride = flatten(stride(lhs))
    return composition(flat_shape, flat_stride, shape(rhs), stride(rhs))
end
@generated function composition(lhs::Layout, rhs::Tuple)
    @assert rank(rhs) <= rank(lhs)
    return :(transform_layout(composition, lhs, rhs, $(static(rank(rhs)))))
end

function composition(lhs::Layout, rhs::Colon)
    return lhs
end
function composition(lhs::Layout, rhs)
    return composition(lhs, make_layout(rhs))
end

"""
    compose(l1::Layout, l2::Layout)

Compose two layouts as composing two functions. You can use `∘` operator as well.

## Examples

```julia
julia> make_layout(20, 2) ∘ make_layout((4, 5), (1, 4))
(4, 5):(2, 8)


julia> make_layout(20, 2) ∘ make_layout((4, 5), (5, 1))
(4, 5):(10, 2)
"""
function compose(l1::Layout, l2::Layout)
    return composition(l1, l2)
end
function compose(l1::Layout, arg1, args...)
    return composition(l1, (arg1, args...))
end

function Base.:(∘)(l1::Layout, l2::Layout)
    return composition(l1, l2)
end
function Base.:(∘)(l1::Layout, args::Vararg{Union{Layout, Colon}, N}) where {N}
    return composition(l1, (args...))
end

function withshape(l::Layout, shape::GenIntTuple)
    return composition(l, make_layout(shape))
end
function withshape(l::Layout, s1, s2, s3...)
    return composition(l, make_layout((s1, s2, s3...)))
end

function _complement_inner(init, i)
    _curr_stride = Static.reduce_tup(min, init[2])
    _curr_idx = static_findfirst(==(_curr_stride), init[2])
    _curr_shape = init[1][_curr_idx]

    return (remove(init[1], _curr_idx), remove(init[2], _curr_idx),
            append(init[3], _curr_stride ÷ init[4][i]),
            append(init[4], _curr_shape * _curr_stride))
end

function _complement(shape::IntType, stride::StaticInt{0}, cosize_hi::IntType)
    return make_layout(cosize_hi)
end
function _complement(shape::IntType, stride::IntType, cosize_hi::IntType)
    #stride == 0 && return make_layout(cosize_hi)
    rest_stride = shape * stride
    return bw_coalesce(One(), (stride,), (One(),), cld(cosize_hi, rest_stride), rest_stride)
end
function _complement(shape::Tuple{IntType}, stride::Tuple{IntType}, cosize_hi::IntType)
    result_stride = tuple(One())
    result_shape = stride
    rest_stride = first(shape) * first(stride)
    return bw_coalesce(One(), result_shape, result_stride, cld(cosize_hi, rest_stride),
                       rest_stride)
end
function _complement(shape::IntTuple{R}, stride::StaticIntTuple{R},
                     cosize_hi::IntType) where {R}
    curr_stride = Static.reduce_tup(min, stride)   # we assume R > 1 here, R == 1 is handled above
    curr_idx = static_findfirst(==(curr_stride), stride)
    curr_shape = shape[curr_idx]

    _result = (remove(shape, curr_idx), remove(stride, curr_idx), tuple(curr_stride),
               tuple(One(), curr_shape * curr_stride))
    result = _foldl(_complement_inner, ntuple(x -> x + 1, Val(R - 2)), _result)
    result_stride = last(result)
    result_shape = append(result[3], result[2][1] ÷ back(result_stride))

    rest_stride = result[1][1] * result[2][1]
    return bw_coalesce(StaticInt{R}(), result_shape, result_stride,
                       cld(cosize_hi, rest_stride), rest_stride)
end

function complement(l::Layout, cosize_hi::IntType)
    filter_layout = filter(l)
    return _complement(shape(filter_layout), stride(filter_layout), cosize_hi)
end

function complement(l::Layout)
    filter_layout = filter(l)
    return _complement(shape(filter_layout), stride(filter_layout), cosize(filter_layout))
end

# need this specialization to avoid type instability
Base.@assume_effects :total function inverse_seq(shape, stride, I::StaticInt)
    length(shape) < I && return ()
    @inbounds next_stride = stride[I] * shape[I]
    if isa(next_stride, StaticInt)
        next_idx = static_findfirst(==(next_stride), stride)
        return inverse_seq(shape, stride, next_idx, I)
    else
        return tuple(I)
    end
end
function inverse_seq(shape, stride, I::StaticInt, I′::StaticInt,
                     Is::Vararg{StaticInt, N}) where {N}
    length(shape) < I && return (I′, Is...)
    @inbounds next_stride = stride[I] * shape[I]
    if isa(next_stride, StaticInt)
        next_idx = static_findfirst(==(next_stride), stride)
        return inverse_seq(shape, stride, next_idx, (I′, Is..., I)...)
    else
        return (I′, Is..., I)
    end
end

@inline right_inverse(x::Colon) = x

"""
    right_inverse(layout::Layout)

Return the right inverse of `layout`, i.e. a layout `layout′` such that `(layout ∘ layout′)(i) == (i)`.
The domain of `layout′` is chosen to be the maximum continues squence of the codomain of `layout`.
"""
function right_inverse(layout::Layout)
    flat_layout = coalesce(layout)
    astride = map(abs, flat_layout.stride)
    next_I = findfirst(Base.Fix1(===, One()), astride)
    isnothing(next_I) && return @Layout(1, 0)

    iseq = inverse_seq(flat_layout.shape, astride, static(next_I))
    isempty(iseq) && return @Layout(1, 0)

    rstride = compact_col_major(flat_layout.shape)
    return make_layout(unwrap(map(Base.Fix1(shape, flat_layout), iseq)),
                       unwrap(map(i -> sign(Base.Fix1(stride, flat_layout)(i)) *
                                       Base.Fix1(getindex, rstride)(i), iseq)))
end

left_inverse(layout::Layout) = right_inverse(make_layout(layout, complement(layout)))
left_inverse(::Colon) = Colon()

function max_common_layout(a::StaticLayout, b::StaticLayout)
    inv_b = right_inverse(b)
    common = coalesce(composition(a, inv_b))

    if stride(common, 1) == One()
        return composition(inv_b, common[1])
    else
        return @Layout 1 0 # no vectorization
    end
end

function max_common_layout(::Layout, ::Layout)
    @inline
    return @Layout 1 0 # no vectorization
end

@inline max_common_vector(a::Layout, b::Layout) = size(max_common_layout(a, b))

function _zip(layout::Layout)
    return make_layout(_zip(shape(layout)), _zip(stride(layout)))
end
# this is equivalent to make_layout(map(make_layout, l1, l2)...)
function _zip(layoutA::Layout, layoutB::Layout)
    return make_layout(_zip(shape(layoutA), shape(layoutB)),
                       _zip(stride(layoutA), stride(layoutB)))
end

@inline function transpose(l::Layout{2})
    return make_layout(l[2], l[1])
end

function tile_unzip(layout::Layout, tile)
    return make_layout(zip2_by(shape(layout), tile), zip2_by(stride(layout), tile))
end

"""
    logical_product(A::Layout, B::Layout)

Compute the logical product of two layouts. Indexing through the first mode of the resulting layout
corresponds to indexing through `A` and indexing through the second mode corresponds to indexing
through `B`.

```julia
julia> tile = @Layout((2, 2), (1, 2));

julia> print_layout(tile)
(static(2), static(2)):(static(1), static(2))
      1   2
    +---+---+
 1  | 1 | 3 |
    +---+---+
 2  | 2 | 4 |
    +---+---+

julia> matrix_of_tiles = @Layout((3, 4), (4, 1));

julia> print_layout(matrix_of_tiles)
(static(3), static(4)):(static(4), static(1))
       1    2    3    4
    +----+----+----+----+
 1  |  1 |  2 |  3 |  4 |
    +----+----+----+----+
 2  |  5 |  6 |  7 |  8 |
    +----+----+----+----+
 3  |  9 | 10 | 11 | 12 |
    +----+----+----+----+

julia> print_layout(logical_product(tile, matrix_of_tiles))
((static(2), static(2)), (static(3), static(4))):((static(1), static(2)), (static(16), static(4)))
       1    2    3    4    5    6    7    8    9   10   11   12
    +----+----+----+----+----+----+----+----+----+----+----+----+
 1  |  1 | 17 | 33 |  5 | 21 | 37 |  9 | 25 | 41 | 13 | 29 | 45 |
    +----+----+----+----+----+----+----+----+----+----+----+----+
 2  |  2 | 18 | 34 |  6 | 22 | 38 | 10 | 26 | 42 | 14 | 30 | 46 |
    +----+----+----+----+----+----+----+----+----+----+----+----+
 3  |  3 | 19 | 35 |  7 | 23 | 39 | 11 | 27 | 43 | 15 | 31 | 47 |
    +----+----+----+----+----+----+----+----+----+----+----+----+
 4  |  4 | 20 | 36 |  8 | 24 | 40 | 12 | 28 | 44 | 16 | 32 | 48 |
    +----+----+----+----+----+----+----+----+----+----+----+----+
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

function zipped_product(layout::Layout, tile)
    return tile_unzip(logical_product(layout, tile), tile)
end

function tiled_product(layout::Layout, tile)
    d = zipped_product(layout, tile)
    return d(:, repeat(:, rank(tile)))
end

"""
    blocked_product(tile::Layout, matrix_of_tiles::Layout, coalesce_result::Bool=false)

Compute the blocked product of two layouts. Indexing through the first mode of the resulting layout
corresponds to indexing through the cartesian product of the first mode of `tile` and the first
mode of `matrix_of_tiles`. Indexing through the second mode is similar. If `coalesce_result` is
true, then the result is coalesced.

```julia
julia> tile = @Layout (2, 2);

julia> matrix_of_tiles = @Layout (3, 4) (4, 1);

julia> print_layout(blocked_product(tile, matrix_of_tiles))
((static(2), static(3)), (static(2), static(4))):((static(1), static(16)), (static(2), static(4)))
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
function blocked_product(block::Layout{N}, layout::Layout{M},
                         coalesce_result::Bool=false) where {N, M}
    R = max(N, M)
    padded_block = append(block, StaticInt{R}())
    padded_layout = append(layout, StaticInt{R}())
    result = logical_product(padded_block, padded_layout)
    @inbounds result = _zip(result[1], result[2])
    coalesce_result && return coalesce(result, repeat(One(), R))
    return result
end

"""
    raked_product(tile::Layout, matrix_of_tiles::Layout, coalesce_result::Bool=false)

The tile is shattered or interleaved with the matrix of tiles.

```julia
julia> tile = @Layout (2, 2) (1, 2);

julia> matrix_of_tiles = @Layout (3, 4) (4, 1);

julia> print_layout(raked_product(tile, matrix_of_tiles))
((static(3), static(2)), (static(4), static(2))):((static(16), static(1)), (static(4), static(2)))
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
    padded_block = append(block, StaticInt{R}())
    padded_layout = append(layout, StaticInt{R}())
    result = logical_product(padded_block, padded_layout)
    @inbounds result = _zip(result[2], result[1])
    coalesce_result && return coalesce(result, repeat(One(), R))
    return result
end

# tile_to_shape

@generated function safe_div(::StaticInt{N}, ::StaticInt{M}) where {N, M}
    R = div(N, M)
    @assert R * M==N "Safe division failed"
    return :($(StaticInt{R}()))
end

@inline safe_div(x::IntType, y::IntType) = div(x, y)

struct Upcast{N<:StaticInt} end

function (::Upcast)(shape::IntType, stride::StaticInt{0})
    @inline
    return make_layout(shape, stride)
end
function (::Upcast{m})(shape::IntType, stride::StaticInt) where m
    @inline
    return make_layout(shape_div(shape, shape_div(m(), abs(stride))), shape_div(stride, m()))
end
function (::Upcast{m})(shape::IntType, stride::Int) where m
    @inline
    return make_layout(shape, safe_div(stride, m()))
end
function (f::Upcast{m})(shape::Tuple, stride::Tuple) where m
    return transform_layout(f, shape, stride)
end
function upcast(layout::Layout, ::StaticInt{M}) where M
    @inline
    return Upcast{StaticInt{M}}()(layout.shape, layout.stride)
end

struct Downcast{N<:StaticInt} end

function (::Downcast{N})(shape::IntType, stride::StaticInt{1}) where N
    @inline
    return make_layout(shape * N(), stride)
end
function (::Downcast{N})(shape::IntType, stride::StaticInt{-1}) where N
    @inline
    return make_layout(shape * N(), stride)
end
function (::Downcast{N})(shape::IntType, stride::StaticInt) where N
    @inline
    return make_layout(shape, stride * N())
end
function (::Downcast{N})(shape::Int, stride::Int) where N
    @inline
    if isone(stride)
        return make_layout(shape * N(), stride)
    else
        return make_layout(shape, stride * N())
    end
end
function (f::Downcast{N})(shape::Tuple, stride::Tuple) where N
    @inline
    return transform_layout(f, shape, stride)
end

function downcast(layout::Layout, ::StaticInt{M}) where M
    @inline
    return Downcast{StaticInt{M}}()(layout.shape, layout.stride)
end

@generated function recast(layout::Layout, ::Type{NewType},
                           ::Type{OldType}) where {NewType, OldType}
    if sizeof(NewType) == sizeof(OldType)
        return :layout
    elseif sizeof(NewType) > sizeof(OldType)
        @assert sizeof(NewType) % sizeof(OldType)==0 "Cannot recast $OldType to $NewType"
        return :(upcast(layout, $(StaticInt{sizeof(NewType) ÷ sizeof(OldType)}())))
    else
        @assert sizeof(OldType) % sizeof(NewType)==0 "Cannot recast $OldType to $NewType"
        return :(downcast(layout, $(StaticInt{sizeof(OldType) ÷ sizeof(NewType)}())))
    end
end

"""
    logical_divide(layout::Layout, tile::Tile)

Gather the elements of `layout` along all modes into blocks according to `tile`.

```julia
julia> raked_prod = @Layout ((3, 2), (4, 2)) ((16, 1), (4, 2));

julia> print_layout(raked_prod)
((static(3), static(2)), (static(4), static(2))):((static(16), static(1)), (static(4), static(2)))
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
(((1, 2), ((3, 1), (1, 1))), ((1, 2), ((4, 1), (1, 1)))):(((48, 1), ((static(16), static(1)), (48, 2))), ((16, 2), ((static(4), static(2)), (16, 4))))
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
function logical_divide(layout::Layout, tile::Tuple)
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

Compute the logical division of `layout` by `tile`, then group the resulting subtiles into the first
mode and the rest into the second mode.

```julia
julia> raked_prod = @Layout ((3, 2), (4, 2)) ((16, 1), (4, 2));

julia> print_layout(raked_prod)
((static(3), static(2)), (static(4), static(2))):((static(16), static(1)), (static(4), static(2)))
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

julia> subtile = (@Layout(2, 3), @Layout(2, 4)); # gather 2 elements with stride 3 along the first mode and 2 elements with stride 4 along the second mode

julia> print_layout(zipped_divide(raked_prod, subtile))
((static(2), static(2)), (static(3), static(4))):((static(1), static(2)), (static(16), static(4)))
       1    2    3    4    5    6    7    8    9   10   11   12
    +----+----+----+----+----+----+----+----+----+----+----+----+
 1  |  1 | 17 | 33 |  5 | 21 | 37 |  9 | 25 | 41 | 13 | 29 | 45 |
    +----+----+----+----+----+----+----+----+----+----+----+----+
 2  |  2 | 18 | 34 |  6 | 22 | 38 | 10 | 26 | 42 | 14 | 30 | 46 |
    +----+----+----+----+----+----+----+----+----+----+----+----+
 3  |  3 | 19 | 35 |  7 | 23 | 39 | 11 | 27 | 43 | 15 | 31 | 47 |
    +----+----+----+----+----+----+----+----+----+----+----+----+
 4  |  4 | 20 | 36 |  8 | 24 | 40 | 12 | 28 | 44 | 16 | 32 | 48 |
    +----+----+----+----+----+----+----+----+----+----+----+----+
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
    return tiled_divide(l1, l2)  # FIXME
end
function tile(l1::Layout, l2::Layout, l3::Layout...)
    return tiled_divide(l1, (l2, l3...))
end

"""
    make_fragment_like(::Layout) -> Layout
    make_fragment_like(T, ::MoYeArray) -> MoYeArray

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
