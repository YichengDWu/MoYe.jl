Base.@propagate_inbounds Base.getindex(@nospecialize(x::Static.IntType), ::StaticInt{1}) = x
Base.@propagate_inbounds Base.getindex(@nospecialize(x::Static.IntType), ::StaticInt{N}) where N = throw(BoundsError(x, N))

# the recursive type definition is tricky to get right, we put Tuple here to represent it.
const IntTuple = Tuple{Vararg{Union{StaticInt, IntSequence, Tuple}}}

Base.@propagate_inbounds function Base.getindex(@nospecialize(x::IntTuple), I1::StaticInt, I2::StaticInt, Is::StaticInt...)
    return getindex(getindex(x, I1), I2, Is...)
end

@inline rank(@nospecialize x::IntTuple) = static(nfields(x))
@inline rank(@nospecialize x::StaticInt) = static(1)
@inline rank(@nospecialize(x::IntTuple), I::StaticInt...) = rank(getindex(x, I...))

function Base.max(@nospecialize(a::StaticInt), @nospecialize(b::StaticInt), @nospecialize(c::StaticInt), @nospecialize(xs::StaticInt...))
    Static.reduce_tup(max, (a, b, c, xs...))
end

function Base.min(@nospecialize(a::StaticInt), @nospecialize(b::StaticInt), @nospecialize(c::StaticInt), @nospecialize(xs::StaticInt...))
    Static.reduce_tup(min, (a, b, c, xs...))
end

# shape

@inline depth(@nospecialize x::StaticInt) = static(0)
function depth(@nospecialize x::IntTuple)
    return max(map(depth, x)...) + static(1)
end


Base.prod(@nospecialize x::IntTuple) = Static.reduce_tup(*, flatten(x))

prod_each(@nospecialize x::IntSequence) = prod(x)
prod_each(@nospecialize x::IntTuple) = map(prod_each2, x)

Base.prod(@nospecialize(x::IntTuple), b::StaticInt, e::StaticInt) = prod(getindex(x, make_int_range(b, e)))

Base.size(@nospecialize x::IntTuple) = prod(x)
Base.size(@nospecialize(x::IntTuple), I::StaticInt, Is::StaticInt) = size(getindex(x, I, Is...))

Base.sum(@nospecialize x::IntTuple) = Static.reduce_tup(+, flatten(x))

inner_product(@nospecialize(x::IntSequence), @nospecialize(y::IntSequence)) = Static.reduce_tup(+, map(*, x, y))
inner_product(@nospecialize(x::IntTuple), @nospecialize(y::IntTuple)) = Static.reduce_tup(+, map(inner_product3, x, y))

Base.cld(@nospecialize(x::IntSequence), @nospecialize(y::IntSequence)) = map(cld, x, y)
function Base.cld(@nospecialize(x::IntTuple), @nospecialize(y::IntTuple))
    @assert rank(x) >= rank(y)
    y = append(y, static(1), rank(x))
    return map(cld, x, y)
end

#shape_div

@inline function elementwise_scale(@nospecialize(x::StaticInt), @nospecialize(y))
    return x * prod(y)
end

function elementwise_scale(@nospecialize(x::IntTuple), @nospecialize(y::IntTuple))
    @assert rank(x) == rank(y)
    return map(elementwise_scale, x, y)
end

function iscongruent(@nospecialize(x::IntTuple), @nospecialize(y::IntTuple))
    return repeat_like(x, static(0)) === repeat_like(y, static(0))
end

# Any coordinate into A can also be used as a coordinate into B
#@inline iscompatiable(@nospecialize(A::StaticInt), @nospecialize(B::IntTuple)) = static(A == size(B))
#@inline iscompatiable(@nospecialize(A::IntTuple), @nospecialize(B::StaticInt)) =false
#function iscompatiable(@nospecialize(A::IntTuple), @nospecialize(B::IntTuple))
#    rank(A) === rank(B) || return  false
#    return Static.reduce_tup(&, map(iscompatiable, A, B))
#end

# Replace the elements of Tuple B that are paired with an Int<0> with an Int<1>
@inline filter_zeros(::StaticInt{0}, @nospecialize(y)) = static(1)
@inline filter_zeros(@nospecialize(::StaticInt), @nospecialize(x)) = x
filter_zeros(@nospecialize(x::IntTuple), @nospecialize(y::IntTuple)) = map(filter_zeros, x, y)
filter_zeros(@nospecialize t::Tuple) = filter_zeros(t, t)

function make_int_tuple(N::StaticInt, t, n::Int, init::Int)
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
    ntuple(static(N)) do i
        @inbounds result[i] = x[i]
    end
    return result
end

# comparison
lex_less(a, b) = a < b
lex_less(@nospecialize(a::Tuple), @nospecialize(b::Tuple)) = lex_less_impl(a, b, static(1))

function lex_less_impl(@nospecialize(a::Tuple), @nospecialize(b::Tuple), @nospecialize(I))
    I > length(b) && return false
    I > length(a) && return true
    return lex_less(getindex(a, I), getindex(b, I)) || (getindex(a, I) == getindex(b, I) && lex_less_impl(a, b, I + static(1)))

    #if I > length(b)
   #     return false
   # elseif I > length(a)
    #    return true
  #  else
   #     return lex_less(a[I], b[I]) || (a[I] == b[I] && lex_less_impl(a, b, I + one(I)))
  #  end
end



# increment

# iterator
