Base.@assume_effects :terminates_locally @generated function coord_to_index0_itt(coord::IntType, shape::Tuple, stride::Tuple) 
    N = length(shape.parameters)
    if N == 1
        if shape.parameters[1] <: Tuple
            return :(coord_to_index0_itt(coord, shape[1], stride[1]))
        else
            return :(coord * stride[1])
        end
    elseif coord == StaticInt{0} 
        expr = Expr(:call, :+,)
        for i in 1:N
            if shape.parameters[i] <: Tuple
                push!(expr.args, :(coord_to_index0_itt(Zero(), shape[$i], stride[$i])))
            else
                push!(expr.args, :(Zero() * stride[$i]))
            end
        end
        return expr
    else
        expr = Expr(:block, :(result = Zero()), :(new_coord = coord))
        for i in 1:N
            if shape.parameters[i] <: Tuple
                push!(expr.args, :(result += coord_to_index0_itt(new_coord % product(shape[$i]), shape[$i], stride[$i])))
            else 
                push!(expr.args, :(result += (new_coord % shape[$i]) * stride[$i]))
            end
            push!(expr.args, :(new_coord = new_coord ÷ product(shape[$i])))
        end

        push!(expr.args, :(return result))
        return expr
    end
end

Base.@assume_effects :total @generated function coord_to_index0_itt2(coord, shape, stride, I0::StaticInt{L}, Is...) where {L}
    if length(Is) == 0 
        if shape.parameters[L] <: Tuple
            return :(coord_to_index0_itt(coord, shape[$L], stride[$L], $(ntuple(static, length(shape.parameters[L].parameters))...)))
        else
            return :(coord * stride[$L])
        end
    elseif coord == StaticInt{0}
        expr = Expr(:call, :+, :(coord_to_index0(Zero(), shape[$L], stride[$L])))
        for i in Is
            push!(expr.args, :(coord_to_index0(Zero(), shape[$(i())], stride[$(i())])))
        end
        return expr
    else
        if shape.parameters[L] <: Tuple
            return quote
                coord_to_index0_itt(coord % product(shape[$L]), shape[$L], stride[$L], $(ntuple(static, length(shape.parameters[L].parameters))...)) +
                coord_to_index0_itt(coord ÷ product(shape[$L]), shape, stride, $(map(x->x(),Is)...))
            end
        else
            return quote
                (coord % product(shape[$L])) * stride[$L] +
                coord_to_index0_itt(coord ÷ product(shape[$L]), shape, stride, $(map(x->x(),Is)...))
            end
        end
    end
end

Base.@assume_effects :total function coord_to_index0(coord::IntType, shape::IntType, stride::IntType)
    @inline
    return coord * stride
end
Base.@assume_effects :total function coord_to_index0(coord::IntType, shape::IntTuple{N}, stride::IntTuple{N}) where {N}
    @inline    return coord_to_index0_itt(coord, shape, stride)
end

@generated function coord_to_index0(coord::NTuple{N, <:StaticInt}, shape::NTuple{N, <:StaticInt},  stride::NTuple{N, <:StaticInt}) where {N}
    coord, stride = make_tuple(coord), make_tuple(stride)
    result = sum((c * s for (c, s) in zip(coord, stride)))
    return :($result)
end
Base.@assume_effects :total @generated function coord_to_index0(coord::IntTuple{N}, shape::IntTuple{N}, stride::IntTuple{N}) where {N}
    expr = Expr(:call, :+)
    for i in 1:N
        push!(expr.args, :(coord_to_index0(coord[$i], shape[$i], stride[$i])))
    end
    return expr
end

Base.@assume_effects :total function coord_to_index0_horner(coord, shape, I1, Is...)
    if isempty(Is)
        return coord[I1]
    else
        return coord[I1] + shape[I1] * coord_to_index0_horner(coord, shape, Is...)
    end
end

Base.@assume_effects :total function coord_to_index0(coord::IntType, shape)
    return coord
end
Base.@assume_effects :total function coord_to_index0(coord::IntTuple{N}, shape::IntTuple{N}) where {N}
    flat_coord = flatten(coord)
    flat_shape = flatten(product_like(shape, coord))
    coord_to_index0_horner(flat_coord, flat_shape, ntuple(identity, rank(flat_shape))...)
end

@inline _offset(x::Colon) = Zero()
@inline _offset(x::Int) = x - one(x)
@inline _offset(x::StaticInt{N}) where {N} = StaticInt{N - 1}()
@inline _offset(x::NTuple{N, Colon}) where {N} = ntuple(Returns(Zero()), Val(N))
@inline function _offset(x::NTuple{N, Int}) where {N}
    return ntuple(Base.Fix2(-, 1) ∘ Base.Fix1(getindex, x), Val(N))
end
@inline _offset(x::Tuple) = map(_offset, x)

Base.@assume_effects :total function coord_to_index(coord::IntType, shape, stride...)
    idx = coord_to_index0(coord - one(coord), shape, stride...)
    return idx + one(idx)
end
Base.@assume_effects :total function coord_to_index(coord, shape, stride...)
    idx = coord_to_index0(_offset(coord), shape, stride...)
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

"""
    GenColMajor

[`make_layout`](@ref) uses this to create a col-major compact layout.
```julia
julia> make_layout(((1, (2, 4)), 1), MoYe.GenColMajor)
((1, (2, 4)), 1):((_1, (1, 2)), 8)
```
"""
const GenColMajor = LayoutLeft

"""
    GenRowMajo

[`make_layout`](@ref) uses this to create a row-major compact layout.
```julia
julia> make_layout(((1, (2, 4)), 1), MoYe.GenRowMajor)
((1, (2, 4)), 1):((8, (4, 1)), _1)
```
"""
const GenRowMajor = LayoutRight

struct CompactLambda{Major} end

function compact_inner_left(init, shape)
    for si in shape.parameters
        current = init[2]
        result = if si == One
            (Zero, current)
        elseif si <: StaticInt
            (current, si * current)
        else # if si <: Tuple
            compact_inner_left((Tuple{}, current), si)
        end
        @inbounds init = (append(init[1], result[1]), result[2])
    end
    return init
end

@generated function compact(shape::StaticIntTuple, current::StaticInt, ::Type{LayoutLeft})
    return :($(map(make_tuple, compact_inner_left((Tuple{}, current), shape))))
end

function compact_inner_right(init, _shape)
    shape = reverse(_shape)
    for si in shape.parameters
        current = init[2]
        result = if si == One
            (Zero, current)
        elseif si <: StaticInt
            (current, si * current)
        else # if si <: Tuple
            compact_inner_right((Tuple{}, current), si)
        end
        @inbounds init = (prepend(init[1], result[1]), result[2])
    end
    return init
end

@generated function compact(shape::StaticIntTuple, current::StaticInt, ::Type{LayoutRight})
    return :($(map(make_tuple, compact_inner_right((Tuple{}, current), shape))))
end

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

function compact_major(shape::Tuple, current::Tuple, ::Type{Major}) where {Major}
    length(shape) == length(current) ||
        throw(DimensionMismatch("shape and current must have the same rank"))
    return map((s, c) -> compact_major(s, c, Major), shape, current)
end
function compact_major(shape, current::IntType, major::Type{Major}) where {Major}
    return @inbounds first(compact(shape, current, major))
end

Base.@assume_effects :total function (::CompactLambda{LayoutLeft})(init, si)
    result = compact(si, init[2], LayoutLeft)
    return @inbounds (append(init[1], result[1]), result[2])
end
Base.@assume_effects :total function (::CompactLambda{LayoutRight})(init, si)
    result = compact(si, init[2], LayoutRight)
    return @inbounds (prepend(init[1], result[1]), result[2])
end

compact_col_major(shape, current=One()) = compact_major(shape, current, LayoutLeft)
compact_row_major(shape, current=One()) = compact_major(shape, current, LayoutRight)

function compact_order(shape::Tuple, order::Tuple, old_shape, old_order)
    return let old_shape = old_shape, old_order = old_order
        map((x, y) -> compact_order(x, y, old_shape, old_order), shape, order)
    end
end
function compact_order(shape, order::StaticInt, old_shape, old_order)
    d = let order = order
        product(map((s, o) -> ifelse(o < order, product(s), One()), old_shape, old_order))
    end
    return compact_col_major(shape, d)
end
function compact_order(shape, order)
    old_shape = flatten_to_tuple(product_like(shape, order))
    flat_order = flatten_to_tuple(order)

    max_order = _foldl(flat_order, Zero()) do v, o
        ifelse(Static.le(v, o) isa Static.True, o, v)
    end    
    old_order = map(ntuple(i->static(i+max_order), Val(rank(flat_order))), flat_order) do seq_v, o
        ifelse(o isa StaticInt, o, seq_v)
    end
    new_order = unflatten(old_order, order)
    return compact_order(shape, new_order, old_shape, old_order)
end
function compact_order(shape, ::Type{LayoutLeft})
    return compact_col_major(shape)
end
function compact_order(shape, ::Type{LayoutRight})
    return compact_row_major(shape)
end
