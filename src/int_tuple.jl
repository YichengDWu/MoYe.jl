# the recursive type definition is tricky to get right, we put Tuple here to represent it.
const IntTuple = Tuple{Vararg{Union{Int, IntSequence, Tuple}}}

Base.@propagate_inbounds function Base.getindex(@nospecialize(x::IntTuple), I1::Int, I2::Int, Is::Int...)
    return getindex(getindex(x, I1), I2, Is...)
end

@inline rank(@nospecialize x::IntTuple) = nfields(x)
@inline rank(@nospecialize x::Int) = 1
@inline rank(@nospecialize(x::IntTuple), I::Int...) = rank(getindex(x, I...))

# shape

@inline depth(@nospecialize x::Int) = static(0)
function depth(@nospecialize x::IntTuple)
    return max(map(depth, x)...) + static(1)
end


Base.prod(@nospecialize x::IntTuple) = Static.reduce_tup(*, flatten(x))

prod_each(@nospecialize x::IntSequence) = prod(x)
prod_each(@nospecialize x::IntTuple) = map(prod_each2, x)

Base.prod(@nospecialize(x::IntTuple), b::Int, e::Int) = prod(getindex(x, make_int_range(b, e)))

Base.size(@nospecialize x::IntTuple) = prod(x)
Base.size(@nospecialize(x::IntTuple), I::Int, Is::Int) = size(getindex(x, I, Is...))

Base.sum(@nospecialize x::IntTuple) = Static.reduce_tup(+, flatten(x))

inner_product(x::IntSequence, y::IntSequence) = Static.reduce_tup(+, map(*, x, y))
inner_product(@nospecialize(x::IntTuple), @nospecialize(y::IntTuple)) = Static.reduce_tup(+, map(inner_product, x, y))

Base.cld(x::IntSequence, y::IntSequence) = map(cld, x, y)
function Base.cld(x::IntTuple, y::IntTuple)
    @assert rank(x) >= rank(y)
    y = append(y, static(1), rank(x))
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
#lex_le = <=
#lex_ge = >=



elem_less_impl(x::Int, y::Int) = x < y
elem_less_impl(::Tuple{}, ::Tuple{}) = true
elem_less_impl(::Tuple{}, ::Tuple) = true #  TupleA is exhausted
elem_less_impl(::Tuple, ::Tuple{}) = false # TupleA is not exhausted, TupleB is exhausted
elem_less_impl(x::Tuple{Int}, y::Tuple{Int}) = x[1] < y[1]

function elem_less_impl(t1::NTuple{N,Int}, t2::NTuple{M,Int}) where {N,M}
    a, b = t1[1], t2[1]
    if !((a == b) || elem_less_impl(a, b))
        return false
    end
    return elem_less_impl(Base.tail(t1), Base.tail(t2))
end

# increment

# iterator
