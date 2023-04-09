apply(f::Function, @nospecialize(t), @nospecialize(I::Tuple)) = f(getindex(t, I)...)
apply(f::Function, @nospecialize(t)) = f(t...)

# (g, f, t1, t2) = g(f(t_1_1, t_2_1), f(t_1_2, t_2,_2),...)
function tapply(g::Function, f::Function, @nospecialize(I::IntSequence), @nospecialize(t::Tuple...))
    return g(broadcast(f, broadcast(Base.Fix2(getindex, I), t)...)...)
end

@inline transform_apply(g::Function, f::Function, t::Tuple...) = g(broadcast(f, t...)...)

function for_each(f::Function, @nospecialize(t))
    f.(t)
    return nothing
end

#for_each_leaf(f::Function, t) = fmap(f, t)

@inline transform(f::Function, @nospecialize(t::Tuple...)) = tuple(broadcast(f, t...)...)

#transform_leaf(f::Function, t) = fmap(f, t)

function Base.findfirst(f::Function, @nospecialize(t::IntSequence), @nospecialize(I::IntSequence))
    return (@inline; f(t[first(I)])) ? first(I) : findfirst(f, t, Base.tail(I))
end
@inline function Base.findfirst(::Function, @nospecialize(t::IntSequence), ::Tuple{})
    return nothing
end
# this would overwrite Base.findfirst(f, t::Tuple) if we don't use IntSequence
Base.findfirst(f::Function, @nospecialize(t::IntSequence)) = findfirst(f, t, tuple_seq(t))
Base.findfirst(@nospecialize(x::StaticInt), @nospecialize(t::IntSequence)) = findfirst(==(x), t)

Base.in(@nospecialize(x::StaticInt), @nospecialize(t::Tuple)) = static(!isnothing(findfirst(==(x), t)))
Base.any(f::Function, @nospecialize(t::IntSequence)) = static(!isnothing(findfirst(f, t)))
Base.all(f::Function, @nospecialize(t::IntSequence)) = static(isnothing(findfirst(!f, t)))


# filter # normal filter
# fold
# fold_first
# front
# back
# take
# unwrap
# flatten_to_tuple # normal flatten
# flatten # normal flatten

# construct
# insert
# remove
# replace
# replace_front
@inline function replace_front(@nospecialize(t::Tuple), @nospecialize(v::StaticInt))
    return (v, Base.tail(t)...)
end

# replace_back

@inline function repeat(@nospecialize(x::StaticInt), @nospecialize(n::StaticInt))
    return ntuple(i -> x, n)
end

# repeat_like
# group
# append
# prepend
# iscan
# escan
# zip
