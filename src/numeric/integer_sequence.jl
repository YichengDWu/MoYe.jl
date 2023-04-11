@inline make_int_sequence(x) = ntuple(i -> i, x)
@inline make_int_range(x::Integer, y::Integer) = (x:y...,)

@inline tuple_seq(@nospecialize t::Tuple) = make_int_sequence(length(t))

const IntSequence{N} = NTuple{N, Int}

Base.@propagate_inbounds Base.getindex(x::Tuple, I::IntSequence{N}) where {N} = map(Base.Fix1(getindex, x), I)
