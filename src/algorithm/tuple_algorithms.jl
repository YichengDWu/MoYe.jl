# we don't overload Base.front, the following finds the first non-tuple element
front(@nospecialize(t::Tuple)) = front(first(t))
@inline front(x) = x

back(@nospecialize(t::Tuple)) = back(getindex(t, length(t)))
@inline back(x) = x

unwrap(@nospecialize(t::Tuple)) = isone(nfields(t)) ? unwrap(first(t)) : t
@inline unwrap(x) = x

# recursive flatten
@inline flatten(::Tuple{}) = ()
flatten(@nospecialize x::Tuple) = (flatten(first(x))..., flatten(Base.tail(x))...)
@inline flatten(x) = x

tuple_cat(x) = x
tuple_cat(x, y, z...) = (x..., tuple_cat(y, z...)...)

function insert(@nospecialize(t::Tuple), x, N)
    return (getindex(t, Base.OneTo(N - one(N)))..., x, getindex(t, N:length(t))...)
end

function remove(t::Tuple, ::StaticInt{N}) where {N}
    M = length(t)
    M < N && return t
    M == N && return Base.front(t)
    if @generated
        f = ntuple(i-> :(x[$i]), N-1)
        t = ntuple(i-> :(x[$(i+N)]), M-N)
        quote
            ($(f...), $(t...))
        end
    else
        return (getindex(t, Base.OneTo(N - one(N)))...,
                getindex(t, UnitRange(N + one(N), length(t)))...)
    end
end

function Base.replace(@nospecialize(t::Tuple), x, N)
    return (getindex(t, Base.OneTo(N - one(N)))..., x,
            getindex(t, UnitRange(N + one(N), length(t)))...)
end

@inline function replace_front(@nospecialize(t::Tuple), v)
    return (v, Base.tail(t)...)
end
@inline replace_front(t, v) = v

@inline function replace_back(@nospecialize(t::Tuple), v)
    return (Base.front(t)..., v)
end
@inline replace_back(t, v) = v

@inline function Base.repeat(x, n)
    return ntuple(i -> x, n)
end

@inline repeat_like(t, x) = x
function repeat_like(t::Tuple, x)
    return map(Base.Fix2(repeat_like, x), t)
end

@generated function repeat_like(::Type{T}, x) where {T<:Tuple}
    expr = Expr(:tuple)
    function repeat_inner(expr, T)
        for i in T.parameters
            if i <: IntType
                push!(expr.args, :x)
            elseif i <: Tuple
                push!(expr.args, repeat_inner(Expr(:tuple), i))
            end
        end
        return expr
    end
    repeat_inner(expr, T)
    return expr
end

# Group the elements [B,E] of a T into a single element
function group(@nospecialize(t::Tuple), b, e)
    return (getindex(t, Base.OneTo(b - one(b)))..., getindex(t, UnitRange(b, e)),
            getindex(t, UnitRange(e + one(e), length(t)))...)
end

# append x to extend t to rank N
@inline function append(t::Union{Tuple, IntType}, val, ::StaticInt{N}) where {N}
    M = length(t)
    M > N && throw(ArgumentError(LazyString("input tuple of length ", M, ", requested ", N)))
    if @generated
        quote
            (t..., $(fill(:val, N - length(t.parameters))...))
        end
    else
        (t..., ntuple(Returns(val), N-M)...)
    end
end
function append(t::Tuple, x)
    @inline
    return (t..., x)
end
function append(t::IntType, x::IntType)
    @inline
    return (t, x)
end

@inline function prepend(t::Union{Tuple, IntType}, val, ::StaticInt{N}) where {N}
    M = length(t)
    M > N && throw(ArgumentError(LazyString("input tuple of length ", M, ", requested ", N)))
    if @generated
        quote
            ($(fill(:val, N - length(t.parameters))...), t...)
        end
    else
        (ntuple(Returns(val), N-M)..., t...)
    end
end
function prepend(t::Tuple, x)
    @inline
    return (x, t...)
end
function prepend(t::IntType, x::IntType)
    @inline
    return (x, t)
end

@generated function escan(f::Function, x::NTuple{N, T}, init::T) where {N, T}
    q = Expr(:block, Expr(:meta, :inline, :propagate_inbounds))
    if N == 1
        push!(q.args, :init)
        return q
    end

    syms = ntuple(i -> Symbol(:i_, i), N)
    push!(q.args, Expr(:(=), syms[1], :init))
    for n in 1:(N - 1)
        push!(q.args, Expr(:(=), syms[n + 1], Expr(:call, :f, syms[n], Expr(:ref, :x, n))))
    end
    push!(q.args, Expr(:return, Expr(:tuple, syms...)))
    return q
end

@inline function _transpose(@nospecialize(t1::Tuple), @nospecialize(t2::Tuple),
                            @nospecialize(ts::Tuple...))
    return tuple(zip(t1, t2, ts...)...)
end

function zip2_by(t, guide::Tuple)
    TR = length(t)
    GR = length(guide)
    GR <= TR || throw(ArgumentError("zip2_by: guide tuple is longer than input tuple"))
    split = map(zip2_by, t[1:GR], guide[1:GR])
    result = tuple(zip(split...)...)
    return (result[1], (result[2]..., t[(GR + 1):end]...))
end
function zip2_by(t, guide)
    return t
end
