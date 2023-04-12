# the recursive type definition is tricky to get right, we put Tuple here to represent it.
const IntSequence{N} = NTuple{N, Int}

Base.@propagate_inbounds Base.getindex(x::Tuple, I::IntSequence{N}) where {N} = map(Base.Fix1(getindex, x), I)

const IntTuple = Tuple{Vararg{Union{Int, IntSequence, Tuple}}}

Base.@propagate_inbounds function Base.getindex(@nospecialize(x::IntTuple), I1::Int, I2::Int, Is::Int...)
    return getindex(getindex(x, I1), I2, Is...)
end

# fmap where leaves are integers
emap(f::Function, t::IntTuple) = map(Base.Fix1(emap, f), t)
emap(f::Function, x::Int) = f(x)

@inline rank(@nospecialize x::IntTuple) = nfields(x)
@inline rank(@nospecialize x::Int) = 1
@inline rank(@nospecialize(x::IntTuple), I::Int...) = rank(getindex(x, I...))

# shape

@inline depth(@nospecialize x::Int) = 0
function depth(@nospecialize x::IntTuple)
    return max(map(depth, x)...) + 1
end


Base.prod(@nospecialize x::IntTuple) = prod(flatten(x))

prod_each(@nospecialize x::IntSequence) = prod(x)
prod_each(@nospecialize x::IntTuple) = map(prod_each2, x)

Base.prod(@nospecialize(x::IntTuple), b::Int, e::Int) = prod(getindex(x, b:e))

Base.size(@nospecialize x::IntTuple) = prod(x)
Base.size(@nospecialize(x::IntTuple), I::Int, Is::Int) = size(getindex(x, I, Is...))

Base.sum(@nospecialize x::IntTuple) = sum(flatten(x))

inner_product(x::IntSequence, y::IntSequence) = sum(map(*, x, y))
inner_product(@nospecialize(x::IntTuple), @nospecialize(y::IntTuple)) = sum(map(inner_product, x, y))

Base.cld(x::IntSequence, y::IntSequence) = map(cld, x, y)
function Base.cld(x::IntTuple, y::IntTuple)
    @assert rank(x) >= rank(y)
    y = append(y, 1, rank(x))
    return map(cld, x, y)
end

#shape_div

@inline function elem_scale(x::Int, y)
    return x * prod(y)
end

function elem_scale(@nospecialize(x::IntTuple), @nospecialize(y::IntTuple))
    @assert rank(x) == rank(y)
    return map(elem_scale, x, y)
end

function iscongruent(@nospecialize(x::IntTuple), @nospecialize(y::IntTuple))
    return repeat_like(x, 0) === repeat_like(y, 0)
end

# Any coordinate into A can also be used as a coordinate into B
#@inline iscompatiable(@nospecialize(A::Int), @nospecialize(B::IntTuple)) = static(A == size(B))
#@inline iscompatiable(@nospecialize(A::IntTuple), @nospecialize(B::Int)) =false
#function iscompatiable(@nospecialize(A::IntTuple), @nospecialize(B::IntTuple))
#    rank(A) === rank(B) || return  false
#    return Static.reduce_tup(&, map(iscompatiable, A, B))
#end

# Replace the elements of Tuple B that are paired with an Int<0> with an Int<1>
@inline filter_zeros(a::Int, x) = iszero(a) ? 1 : x
filter_zeros(@nospecialize(x::IntTuple), @nospecialize(y::IntTuple)) = map(filter_zeros, x, y)
filter_zeros(@nospecialize t::Tuple) = filter_zeros(t, t)

function make_int_tuple(N::Int, t, n::Int, init::Int)
    ntuple(N) do i
        i ≤ n ? t[i] : init
    end
end

# fill_int_tuple_from

# make_int_tuple_from

function to_array(::Type{T}, @nospecialize(x::IntTuple)) where T
    x = flatten(x)
    N = length(x)
    result = Array{T}(undef, N)
    ntuple(N) do i
        @inbounds result[i] = x[i]
    end
    return result
end

# comparison
# Base.:(<)(x::Int, y::Tuple) = x < prod(y) # maybe we need this for non congruent shapes
#lex_less = <
#lex_leq = <=
#lex_geq = >=

colex_less(x::Int, y::Int) = x < y
colex_less(::Tuple{}, ::Tuple{}) = false
colex_less(::Tuple{}, ::Tuple) = true
colex_less(::Tuple, ::Tuple{}) = false
function colex_less(t1::Tuple, t2::Tuple)
    a, b = last(t1), last(t2)
    if a ≠ b
        return colex_less(a, b)
    end
    return colex_less(Base.front(t1), Base.front(t2))
end

elem_less(x::Int, y::Int) = x < y
elem_less(::Tuple{}, ::Tuple{}) = true
elem_less(::Tuple{}, ::Tuple) = true #  TupleA is exhausted
elem_less(::Tuple, ::Tuple{}) = false # TupleA is not exhausted, TupleB is exhausted

function elem_less(t1::Tuple, t2::Tuple)
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

increment(coord::Int, shape::Int) = ifelse(coord < shape, coord + 1, 1)
function increment(coord::Coord, shape::Shape) where {Coord, Shape}
    c, s = first(coord), first(shape)
    if length(coord) == length(shape) == 1
        return increment(c, s)
    end

    if c != s
        return (increment(c, s), Base.tail(coord)...)
    end
    return (repeat_like(s, 1), increment(Base.tail(coord), Base.tail(shape))...)
 end

 # iterator

struct ForwardCoordUnitRange{B, E} <:AbstractUnitRange{Int}
    start::B
    stop::E

    function ForwardCoordUnitRange(start::IntTuple, stop::IntTuple)
        new{typeof(start), typeof(stop)}(start, stop)
    end
end

const ForwardCoordOneTo{T} = ForwardCoordUnitRange{T, T}
function ForwardCoordOneTo(shape::IntTuple)
    start = repeat_like(shape, 1)
    ForwardCoordUnitRange(start, shape)
end

Base.oneto(shape::IntTuple) = ForwardCoordOneTo(shape)
Base.first(x::ForwardCoordOneTo) = getfield(x, :start)
Base.last(x::ForwardCoordOneTo) = getfield(x, :stop)
Base.length(x::ForwardCoordOneTo) = length(getfield(x, :stop))

function Base.iterate(x::ForwardCoordOneTo)
    start = getfield(x, :start)
    return (start, start)
end
function Base.iterate(x::ForwardCoordOneTo, state)
    stop = getfield(x, :stop)
    if state == stop
        return nothing
    end
    new_state = increment(state, stop)
    return (new_state, new_state)
end
