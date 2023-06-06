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

@generated function remove(x::Tuple, ::StaticInt{N}) where {N}
    M = length(x.parameters)
    M < N && return :x
    M == N && return :(Base.front(x))

    f = ntuple(i-> :(x[$i]), N-1)
    t = ntuple(i-> :(x[$(i+N)]), M-N)
    return quote
        ($(f...), $(t...))
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
@generated function replace_front(::Type{T}, ::Type{V}) where {T<:Tuple,V}
    expr = Expr(:curly, Tuple)
    push!(expr.args, V)
    push!(expr.args, T.parameters[2:end]...)
    return expr
end

@inline function replace_back(@nospecialize(t::Tuple), v)
    return (Base.front(t)..., v)
end
@inline replace_back(t, v) = v

@inline function Base.repeat(x, n)
    return ntuple(Returns(x), n)
end

@inline repeat_like(t, x) = x
@inline function repeat_like(t::Tuple, x)
    return repeat_like(typeof(t), x)
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


@generated function group(t::Tuple, ::StaticInt{B}, ::StaticInt{E}) where {B,E}
    return quote
        @inbounds (t[1:$B-1]..., t[$B:$E], t[$E+1:end]...)
    end
end

# Group the elements [B,E] of a T into a single element
function group(t::Tuple, b, e)
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
@generated function append(::Type{T}, ::Type{X}) where {T<:Tuple, X}
    expr = Expr(:curly, Tuple)
    push!(expr.args, T.parameters...)
    push!(expr.args, X)
    return expr
end
@generated function append(::Type{T}, ::Type{X}) where {T, X}
    expr = Expr(:curly, Tuple)
    push!(expr.args, T)
    push!(expr.args, X)
    return expr
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
@generated function prepend(::Type{T}, ::Type{X}) where {T<:Tuple, X}
    expr = Expr(:curly, Tuple)
    push!(expr.args, X)
    push!(expr.args, T.parameters...)
    return expr
end
@generated function prepend(::Type{T}, ::Type{X}) where {T, X}
    expr = Expr(:curly, Tuple)
    push!(expr.args, X)
    push!(expr.args, T)
    return expr
end
function prepend(t::IntType, x::IntType)
    @inline
    return (x, t)
end

# specialize on the operation
Base.@assume_effects :total @generated function _foldl(op::G, x::Tuple, init) where {G}
    length(x.parameters) == 0 && return :init
    expr = :(op(init, x[1]))
    for i in 2:length(x.parameters)
        expr = :(op($expr, x[$i]))
    end
    return quote
        Base.@_inline_meta
        @inbounds $expr
    end
end

Base.@assume_effects :total @generated function escan(f::F, x::NTuple{N, T}, init::T) where {F, N, T}
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

@inline _zip(t::Tuple{Vararg{Tuple}}) = tuple(zip(t...)...)
@inline _zip(t::Tuple) = tuple(t)
@inline _zip(t) = t
@inline _zip(t1, t2, t3...) = _zip((t1, t2, t3...))

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

@generated hascolon(::T) where T = :($(Colon ∈ T.parameters))
@generated hascolon(::Type{T}) where T = :($(Colon ∈ T.parameters))
