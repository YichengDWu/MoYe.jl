using Base.Broadcast: AbstractArrayStyle, DefaultArrayStyle, Broadcasted
using Base.Broadcast: _broadcast_getindex, broadcast_shape
import Base.Broadcast: instantiate, materialize!, BroadcastStyle, broadcasted

struct MoYeArrayStyle{N, S} <: AbstractArrayStyle{N} end

BroadcastStyle(::Type{<:StaticMoYeArray{T,N,E,L}}) where {T,N,E,S,L<:Layout{N,S}} = MoYeArrayStyle{N,S}()
BroadcastStyle(a::MoYeArrayStyle, ::DefaultArrayStyle) = a

BroadcastStyle(a::MoYeArrayStyle{N,S}, b::MoYeArrayStyle{N,S}) where {N,S} = a
@generated function BroadcastStyle(a::MoYeArrayStyle{N1,S1}, b::MoYeArrayStyle{N2,S2}) where {N1, S1, N2, S2}
    if N1 > N2
        return :a
    elseif N1 < N2
        return :b
    else
        throw("Don't know how to broadcast MoYeArrays")
    end
end

# currently only defined for static size
@inline function Base.similar(bc::Broadcasted{MoYeArrayStyle{N,S}}, ::Type{ElType}) where {ElType, N, S}
    return MoYeArray{ElType}(undef, make_tuple(S))
end

@inline function Base.copyto!(dest::StaticOwningArray{T,N,L}, bc::Broadcasted{MoYeArrayStyle{N,S}}) where {T,N,S,L<:Layout{N,S}}
    @gc_preserve copyto!(dest, bc)
    return dest
end

# have to define these to avoid ambiguities...
@inline broadcasted(f::Function, x::StaticOwningArray) = @gc_preserve broadcasted(f, x)
@inline broadcasted(f::Function, x::StaticOwningArray, y::StaticOwningArray) = @gc_preserve broadcasted(f, x, y)
@inline broadcasted(f::Function, x::StaticOwningArray, y::Number) = @gc_preserve broadcasted(f, x, y)
@inline broadcasted(f::Function, x::Number, y::StaticOwningArray) = @gc_preserve broadcasted(f, x, y)
@inline broadcasted(f::Function, x::StaticOwningArray, y::StaticNonOwningArray) = @gc_preserve broadcasted(f, x, y)
@inline broadcasted(f::Function, x::StaticNonOwningArray, y::StaticOwningArray) = @gc_preserve broadcasted(f, x, y)

#=
@inline broadcast_sizes(a, as...) = (broadcast_size(a), broadcast_sizes(as...)...)
@inline broadcast_size(a::StaticMoYeArray) = static_size(a)
@inline broadcast_size(a::Tuple) = tuple(static(length(a)))
@inline broadcast_size(a) = ()
@inline broadcast_sizes() = ()

broadcast_getindex(::Tuple{}, i::Int, I::CartesianIndex) = return :(_broadcast_getindex(a[$i], $I))
function broadcast_getindex(oldsize::Tuple, i::Int, newindex::CartesianIndex)
    li = LinearIndices(oldsize)
    ind = _broadcast_getindex(li, newindex)
    return :(a[$i][$ind])
end
@inline function Base.copyto!(dest::StaticMoYeArray{T,N,A,L}, bc::Broadcasted{MoYeArrayStyle{N,S}}) where {T,N,S, A<:ViewEngine,L<:Layout{N,S}}
    flat = broadcast_flatten(bc)
    argsizes = broadcast_sizes(flat.args...)
    @boundscheck axes(dest) == axes(bc) || Broadcast.throwdm(axes(dest), axes(bc))
    return _broadcast!(flat.f, static_size(dest), dest, argsizes, flat.args...)
end

function _broadcast!(f, newsize::NTuple{N, StaticInt}, dest::StaticMoYeArray, argsizes::Tuple, a...) where {N}
    sizes = [dynamic(make_tuple(typeof(sz))) for sz in argsizes]

    indices = CartesianIndices(dynamic(make_tuple(typeof(newsize))))
    exprs = similar(indices, Expr)
    for (j, current_ind) ∈ enumerate(indices)
        exprs_vals = (broadcast_getindex(sz, i, current_ind) for (i, sz) in enumerate(sizes))
        exprs[j] = :(dest[$j] = f($(exprs_vals...)))
    end

    return quote
        Base.@_inline_meta
        @inbounds $(Expr(:block, exprs...))
        return dest
    end
end

# from https://github.com/JuliaLang/julia/pull/43322
function broadcast_flatten(bc::Broadcasted{Style}) where {Style}
    isflat(bc) && return bc
    args = cat_nested(bc)
    makeargs = make_makeargs(bc.args)
    f = Base.maybeconstructor(bc.f)
    newf = (args...) -> (@inline; f(prepare_args(makeargs, args)...))
    return Broadcasted{Style}(newf, args, bc.axes)
end

const NestedTuple = Tuple{<:Broadcasted,Vararg{Any}}
isflat(bc::Broadcasted) = _isflat(bc.args)
_isflat(args::NestedTuple) = false
_isflat(args::Tuple) = _isflat(Base.tail(args))
_isflat(args::Tuple{}) = true

cat_nested(bc::Broadcasted) = cat_nested_args(bc.args)
cat_nested_args(::Tuple{}) = ()
cat_nested_args(t::Tuple{Any}) = cat_nested(t[1])
cat_nested_args(t::Tuple) = (cat_nested(t[1])..., cat_nested_args(tail(t))...)
cat_nested(a) = (a,)

make_makeargs(args::Tuple) = _make_makeargs(args, 1)[1]

@inline function _make_makeargs(args::Tuple, n::Int)
    head, n = _make_makeargs1(args[1], n)
    rest, n = _make_makeargs(tail(args), n)
    (head, rest...), n
end
_make_makeargs(::Tuple{}, n::Int) = (), n

struct Pick{N} <: Function end
(::Pick{N})(@nospecialize(args::Tuple)) where {N} = args[N]

@inline _make_makeargs1(_, n::Int) = Pick{n}(), n + 1
@inline function _make_makeargs1(bc::Broadcasted, n::Int)
    makeargs, n = _make_makeargs(bc.args, n)
    f = Base.maybeconstructor(bc.f)
    makeargs1 = (args::Tuple) -> (@inline; f(prepare_args(makeargs, args)...))
    makeargs1, n
end

@inline prepare_args(makeargs::Tuple, @nospecialize(x::Tuple)) = (makeargs[1](x), prepare_args(tail(makeargs), x)...)
@inline prepare_args(makeargs::Tuple{Any}, @nospecialize(x::Tuple)) = (makeargs[1](x),)
prepare_args(::Tuple{}, ::Tuple) = ()
=#
