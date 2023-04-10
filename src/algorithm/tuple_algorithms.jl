#=
apply(f::Function, @nospecialize(t), @nospecialize(I::Tuple)) = f(getindex(t, I)...)
apply(f::Function, @nospecialize(t)) = f(t...)

# (g, f, t1, t2) = g(f(t_1_1, t_2_1), f(t_1_2, t_2,_2),...)
function tapply(g::Function, f::Function, @nospecialize(I::IntSequence), @nospecialize(t::Tuple...))
    return g(map(f, map(Base.Fix2(getindex, I), t)...)...)
end

@inline transform_apply(g::Function, f::Function, t::Tuple...) = g(map(f, t...)...)

# Iterators.foreach

@inline transform(f::Function, @nospecialize(t::Tuple...)) = tuple(map(f, t...)...)

transform_leaf(f::Function, t) = fmap(f, t)
=#
#=
function Base.findfirst(f::Function, @nospecialize(t::IntSequence), @nospecialize(I::IntSequence))
    return (@inline; f(t[first(I)])) ? first(I) : findfirst(f, t, Base.tail(I))
end
@inline function Base.findfirst(::Function, @nospecialize(t::IntSequence), ::Tuple{})
    return nothing
end
# It would overwrite Base.findfirst(f, t::Tuple) if we don't use IntSequence
Base.findfirst(f::Function, @nospecialize(t::IntSequence)) = findfirst(f, t, tuple_seq(t))
Base.findfirst(@nospecialize(x::StaticInt), @nospecialize(t::IntSequence)) = findfirst(==(x), t)
=#

# maybe we don't need the following functions
#Base.in(@nospecialize(x::StaticInt), @nospecialize(t::Tuple)) = static(!isnothing(findfirst(==(x), t)))
#Base.any(f::Function, @nospecialize(t::IntSequence)) = static(!isnothing(findfirst(f, t)))
#Base.all(f::Function, @nospecialize(t::IntSequence)) = static(isnothing(findfirst(!f, t)))
#  Iterators.filter

# we don't overload Base.front, it finds the first non-tuple element
front(@nospecialize(t::Tuple)) = front(first(t))
@inline front(@nospecialize(x)) = x

back(@nospecialize(t::Tuple)) = back(getindex(t, tuple_size(t)))
@inline back(@nospecialize(x)) = x

# take Takes the elements in the range [B,E] of the tuple
function take(@nospecialize(t::Tuple), B::StaticInt, E::StaticInt)
    return getindex(t, make_int_range(B, E))
end

unwrap(@nospecialize(t::Tuple)) = nfields(t) == 1 ? unwrap(first(t)) : t
@inline unwrap(@nospecialize(x)) = x

# recursive flatten
@inline flatten(::Tuple{}) = ()
flatten(@nospecialize x::Tuple) = (flatten(first(x))..., flatten(Base.tail(x))...)
@inline flatten(@nospecialize(x)) = x

function insert(@nospecialize(t::Tuple), @nospecialize(x), N)
    return (getindex(t, make_int_sequence(N-one(N)))..., x, getindex(t, make_int_range(N, tuple_size(t)))...)
end

function remove(@nospecialize(t::Tuple), N::StaticInt)
    return (getindex(t, make_int_sequence(N-one(N)))..., getindex(t, make_int_range(N+one(N), tuple_size(t)))...)
end

function Base.replace(@nospecialize(t::Tuple), @nospecialize(x), N::StaticInt)
    return (getindex(t, make_int_sequence(N-one(N)))..., x, getindex(t, make_int_range(N+one(N), tuple_size(t)))...)
end

@inline function replace_front(@nospecialize(t::Tuple), @nospecialize(v::StaticInt))
    return (v, Base.tail(t)...)
end

@inline function replace_back(@nospecialize(t::Tuple), @nospecialize(v::StaticInt))
    return (Base.front(t)..., v)
end

@inline function Base.repeat(@nospecialize(x::StaticInt), @nospecialize(n::StaticInt))
    return ntuple(i -> x, n)
end

@inline repeat_like(@nospecialize(t::StaticInt), @nospecialize(x::StaticInt)) = x
function repeat_like(@nospecialize(t::Tuple), @nospecialize(x::StaticInt))
    map(Base.Fix2(repeat_like, x), t)
end

# Group the elements [B,E] of a T into a single element
function group(@nospecialize(t::Tuple), b::StaticInt, e::StaticInt)
    return (getindex(t, make_int_sequence(b-one(b)))..., getindex(t, make_int_range(b, e)), getindex(t, make_int_range(e+one(e), tuple_size(t)))...)
end

# append x to extend t to rank N
function append(@nospecialize(t::Tuple), @nospecialize(x), @nospecialize(I::StaticInt))
    return (t..., ntuple(_ -> x, tuple_size(t)-I)...)
end
function append(@nospecialize(t::Tuple), @nospecialize(x))
    return (t..., x)
end

function prepend(@nospecialize(t::Tuple), @nospecialize(x), @nospecialize(I::StaticInt))
    return (ntuple(_ -> x, tuple_size(t)-I)..., t...)
end
function prepend(@nospecialize(t::Tuple), @nospecialize(x))
    return (x, t...)
end

iscan(f::Function, @nospecialize(x::Tuple), @nospecialize(init = Base._InitialValue())) = (Iterators.accumulate(f, x; init=init)...,)

# escan

@inline function Base.transpose(@nospecialize(t1::Tuple), @nospecialize(t2::Tuple), @nospecialize(ts::Tuple...))
    return (zip(t1, t2, ts...)...,)
end
