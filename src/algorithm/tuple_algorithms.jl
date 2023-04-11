# we don't overload Base.front, it finds the first non-tuple element
front(@nospecialize(t::Tuple)) = front(first(t))
@inline front(x) = x

back(@nospecialize(t::Tuple)) = back(getindex(t, length(t)))
@inline back(x) = x

# take Takes the elements in the range [B,E] of the tuple
function take(@nospecialize(t::Tuple), B, E)
    return getindex(t, make_int_range(B, E))
end

unwrap(@nospecialize(t::Tuple)) = nfields(t) == 1 ? unwrap(first(t)) : t
@inline unwrap(x) = x

# recursive flatten
@inline flatten(::Tuple{}) = ()
flatten(@nospecialize x::Tuple) = (flatten(first(x))..., flatten(Base.tail(x))...)
@inline flatten(x) = x

function insert(@nospecialize(t::Tuple), x, N)
    return (getindex(t, make_int_sequence(N-one(N)))..., x, getindex(t, make_int_range(N, length(t)))...)
end

function remove(@nospecialize(t::Tuple), N)
    return (getindex(t, make_int_sequence(N-one(N)))..., getindex(t, make_int_range(N+one(N), length(t)))...)
end

function Base.replace(@nospecialize(t::Tuple), x, N)
    return (getindex(t, make_int_sequence(N-one(N)))..., x, getindex(t, make_int_range(N+one(N), length(t)))...)
end

@inline function replace_front(@nospecialize(t::Tuple), v)
    return (v, Base.tail(t)...)
end

@inline replace_back(t, v) = v
@inline function replace_back(@nospecialize(t::Tuple), v)
    return (Base.front(t)..., v)
end

@inline function Base.repeat(x, n)
    return ntuple(i -> x, n)
end

@inline repeat_like(t, x) = x
function repeat_like(@nospecialize(t::Tuple), x)
    map(Base.Fix2(repeat_like, x), t)
end

# Group the elements [B,E] of a T into a single element
function group(@nospecialize(t::Tuple), b, e)
    return (getindex(t, make_int_sequence(b-one(b)))..., getindex(t, make_int_range(b, e)), getindex(t, make_int_range(e+one(e), length(t)))...)
end

# append x to extend t to rank N
function append(@nospecialize(t::Tuple), x, I)
    return (t..., ntuple(_ -> x, I-length(t))...)
end
function append(@nospecialize(t::Tuple), x)
    return (t..., x)
end

function prepend(@nospecialize(t::Tuple), x, I)
    return (ntuple(_ -> x, I-length(t))..., t...)
end
function prepend(@nospecialize(t::Tuple), x)
    return (x, t...)
end

iscan(f::Function, @nospecialize(x::Tuple), init = Base._InitialValue()) = (Iterators.accumulate(f, x; init=init)...,)

# escan

@inline function Base.transpose(@nospecialize(t1::Tuple), @nospecialize(t2::Tuple), @nospecialize(ts::Tuple...))
    return (zip(t1, t2, ts...)...,)
end
