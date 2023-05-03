const IntSequence{N} = NTuple{N, Union{Int, StaticInt}}

# the recursive type definition is tricky to get right, we put Tuple here to represent it.
const IntTuple{N} = Tuple{Vararg{Union{Int, StaticInt, Tuple}, N}}
const GenIntTuple = Union{Int, StaticInt, IntTuple}

# note that this type is only **almost** static
const StaticIntTuple{N} = Tuple{
                                Vararg{
                                       Union{StaticInt,
                                             Tuple{Vararg{Union{StaticInt, Tuple}}}}, N}}

Base.@propagate_inbounds function Base.getindex(@nospecialize(x::Tuple),
                                                @nospecialize(I::IntSequence{N})) where {N}
    return map(Base.Fix1(getindex, x), I)
end
# fmap where leaves are integers or colons
emap(f::Function, @nospecialize(t::Tuple)) = map(Base.Fix1(emap, f), t)
emap(f::Function, x::Union{IntType, Colon}) = f(x)

@inline rank(@nospecialize x::IntTuple) = nfields(x)
@inline rank(@nospecialize x::IntType) = one(x)
@inline rank(@nospecialize(x::IntTuple), I::IntType...) = rank(getindex(x, I...))

# shape

@inline depth(@nospecialize x::IntType) = zero(x)
function depth(@nospecialize x::IntTuple)
    return max(map(depth, x)...) + One()
end

@inline product(x::IntType) = x
@inline product(@nospecialize x::IntSequence) = prod(x)
@inline product(@nospecialize x::IntTuple) = prod(flatten(x))

@inline prod_each(@nospecialize x::IntSequence) = prod(x)
@inline prod_each(@nospecialize x::IntTuple) = map(prod_each, x)

@inline capacity(x::IntType) = x
@inline capacity(@nospecialize x::IntTuple) = product(x)
#capacity(@nospecialize(x::IntTuple), I::Int, Is::Int) = capacity(getindex(x, I, Is...))

@inline flatsum(@nospecialize x::IntTuple) = sum(flatten(x))

function inner_product(@nospecialize(x::IntSequence), @nospecialize(y::IntSequence))
    return sum(map(*, x, y))
end
function inner_product(@nospecialize(x::IntTuple), @nospecialize(y::IntTuple))
    return sum(map(inner_product, x, y))
end

Base.cld(@nospecialize(x::IntSequence), @nospecialize(y::IntSequence)) = map(cld, x, y)
function Base.cld(x::IntTuple, y::IntTuple)
    @assert rank(x) >= rank(y)
    y = append(y, One(), StaticInt{rank(x)}())
    return map(cld, x, y)
end

#shape_div
function shape_div(a::IntType, b::IntType)
    return a ÷ b != 0 ? a ÷ b : sign(a) * sign(b)
end
function shape_div(@nospecialize(a::IntType), @nospecialize(b::IntTuple))
    return shape_div(a, product(b))
end
function shape_div(@nospecialize(a::IntTuple), b::IntType)
    result, _ = _foldl((init, ai) -> (append(init[1], shape_div(ai, init[2])),
                                      shape_div(init[2], ai)), a, ((), b))
    return result
end
function shape_div(@nospecialize(a::IntTuple), @nospecialize(b::IntTuple))
    length(a) == length(b) ||
        throw(DimensionMismatch("Tuple A and B must have the same rank"))
    return map(shape_div, a, b)
end

function elem_scale(@nospecialize(x::IntType), @nospecialize(y))
    @inline
    return x * product(y)
end
function elem_scale(@nospecialize(x::IntTuple), @nospecialize(y::IntTuple))
    @assert rank(x) == rank(y)
    return map(elem_scale, x, y)
end

function iscongruent(x, y)
    return repeat_like(typeof(x), 0) === repeat_like(typeof(y), 0)
end

# Any coordinate into A can also be used as a coordinate into B
@inline function iscompatible(@nospecialize(a::Tuple), @nospecialize(b::Tuple))
    return length(a)==length(b) && all(map(iscompatible, a, b))
end
@inline iscompatible(a::IntType, b::GenIntTuple) = a == capacity(b)
@inline iscompatible(a::Tuple, b::IntType) = false

# Replace the elements of Tuple B that are paired with 0 in A with 1
@inline filter_zeros(a::IntType, x) = iszero(a) ? One() : x
function filter_zeros(@nospecialize(x::IntTuple), @nospecialize(y::IntTuple))
    return map(filter_zeros, x, y)
end
filter_zeros(@nospecialize t::Tuple) = filter_zeros(t, t)

function slice(@nospecialize(A::Tuple), @nospecialize(index::Tuple))
    length(A) == length(index) ||
        throw(DimensionMismatch("Array and index must have the same rank"))
    return tuple_cat(map(slice, A, index)...)
end
function slice(A, index::Colon)
    @inline
    return tuple(A)
end
function slice(A, index::IntType)
    @inline
    return ()
end

function dice(@nospecialize(A::Tuple), @nospecialize(index::Tuple))
    length(A) == length(index) ||
        throw(DimensionMismatch("Array and index must have the same rank"))
    return tuple_cat(map(dice, A, index)...)
end
function dice(A, index::Colon)
    @inline
    return ()
end
function dice(A, index::IntType)
    @inline
    return tuple(A)
end

function make_int_tuple(N::IntType, t, n::IntType, init::IntType)
    ntuple(N) do i
        return i ≤ n ? t[i] : init
    end
end

# fill_int_tuple_from

# make_int_tuple_from

function to_array(::Type{T}, @nospecialize(x::IntTuple)) where {T}
    x = flatten(x)
    N = length(x)
    result = Array{T}(undef, N)
    ntuple(N) do i
        @inbounds result[i] = x[i]
    end
    return result
end

# comparison
# Base.:(<)(x::Int, y::Tuple) = x < product(y)?  maybe we need this for non congruent shapes
#lex_less = <
#lex_leq = <=
#lex_geq = >=

colex_less(x::IntType, y::IntType) = x < y
colex_less(::Tuple{}, ::Tuple{}) = false
colex_less(::Tuple{}, ::Tuple) = true
colex_less(::Tuple, ::Tuple{}) = false
function colex_less(@nospecialize(t1::Tuple), @nospecialize(t2::Tuple))
    a, b = last(t1), last(t2)
    if a ≠ b
        return colex_less(a, b)
    end
    return colex_less(Base.front(t1), Base.front(t2))
end

elem_less(x::IntType, y::IntType) = x < y
elem_less(::Tuple{}, ::Tuple{}) = true
elem_less(::Tuple{}, ::Tuple) = true #  TupleA is exhausted
elem_less(::Tuple, ::Tuple{}) = false # TupleA is not exhausted, TupleB is exhausted

function elem_less(@nospecialize(t1::Tuple), @nospecialize(t2::Tuple))
    a, b = first(t1), first(t2)
    if length(t1) == length(t2) == 1
        return a < b
    end

    if !((a == b) || elem_less(a, b))
        return false
    end

    return elem_less(Base.tail(t1), Base.tail(t2))
end

elem_leq(x, y) = !elem_less(y, x)
elem_gtr(x, y) = elem_less(y, x)
elem_geq(x, y) = !elem_geq(x, y)

@inline function increment(coord::IntType, shape::IntType)
    return ifelse(coord < shape, coord + one(coord), one(coord))
end
function increment(coord, shape)
    c, s = first(coord), first(shape)
    if length(coord) == length(shape) == 1
        return increment(c, s)
    end

    if c != s
        return (increment(c, s), Base.tail(coord)...)
    end
    return (repeat_like(typeof(s), 1), increment(Base.tail(coord), Base.tail(shape))...)
end

# iterator

struct ForwardCoordUnitRange{N, B, E} <: AbstractUnitRange{Int}
    start::B
    stop::E

    function ForwardCoordUnitRange(start::IntTuple{N}, stop::IntTuple{N}) where {N}
        return new{N, typeof(start), typeof(stop)}(start, stop)
    end
end

function ForwardCoordOneTo(shape::IntTuple)
    start = repeat_like(typeof(shape), 1)
    return ForwardCoordUnitRange(start, shape)
end

Base.oneto(shape::IntTuple) = ForwardCoordOneTo(shape)
Base.first(x::ForwardCoordUnitRange) = getfield(x, :start)
Base.last(x::ForwardCoordUnitRange) = getfield(x, :stop)
Base.length(x::ForwardCoordUnitRange) = length(getfield(x, :stop))

function Base.iterate(x::ForwardCoordUnitRange)
    start = getfield(x, :start)
    return (start, start)
end
function Base.iterate(x::ForwardCoordUnitRange, state)
    stop = getfield(x, :stop)
    if state == stop
        return nothing
    end
    new_state = increment(state, stop)
    return (new_state, new_state)
end

@generated make_tuple(::Type{StaticInt{N}}) where {N} = StaticInt{N}()
@generated function make_tuple(::Type{S}) where {S<:StaticIntTuple}
    expr = Expr(:tuple)
    for p in S.parameters
        push!(expr.args, make_tuple(p))
    end
    return expr
end

@inline Base.:(==)(::StaticInt{N}, ::StaticInt{N}) where {N} = true
@inline Base.:(==)(@nospecialize(x::StaticInt), @nospecialize(y::StaticInt)) = false

function Base.getindex(x::StaticInt, i::Integer)
    @inline
    @boundscheck i == 1 || throw(BoundsError(x, i))
    x
end
@inline Base.getindex(x::StaticInt, ::StaticInt{1}) = x
@inline Base.getindex(x::StaticInt, ::StaticInt{N}) where {N} = throw(BoundsError(x, N))

# static findfirst, returns StaticInt, returns N+1 if not found, specialized on `f`
function static_findfirst(f::G, t::IntSequence, I::IntSequence) where {G}
    return (@inline; f(t[first(I)])) ? first(I) : static_findfirst(f, t, Base.tail(I))
end
@inline function static_findfirst(::Function, t::IntSequence{N}, ::Tuple{}) where {N}
    return static(N+1)
end
static_findfirst(f::G, t::IntSequence{N}) where {G,N} = static_findfirst(f, t, ntuple(static, N))
static_findfirst(f::G, t::StaticInt) where {G} = ifelse(f(t), One(), static(2))
