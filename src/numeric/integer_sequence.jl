@inline make_int_sequence(@nospecialize x) = ntuple(i -> i, x)
#@inline @generated function make_int_range(::StaticInt{N}, ::StaticInt{M}) where {N, M}
 #   expr = Expr(:tuple)
  #  for i = N:M
   #     push!(expr.args, :(static($i)))
   # end
    #return expr
#end
@inline make_int_range(x::Integer, y::Integer) = (x:y...,)
@inline function make_int_range(@nospecialize(x::StaticInt), @nospecialize(y::StaticInt))
    static((x:y...,))
end

@inline tuple_size(@nospecialize t::Tuple) = nfields(t)
@inline tuple_seq(@nospecialize t::Tuple) = make_int_sequence(tuple_size(t))

const IntSequence{N} = NTuple{N, StaticInt}

Base.@propagate_inbounds Base.getindex(x::Tuple, I::IntSequence{N}) where {N} = broadcast(Base.Fix1(getindex, x), I)
