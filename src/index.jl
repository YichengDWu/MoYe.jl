#=struct  HierarchicalIndex{N, S}
    I::IntTuple{N}
    HierarchicalIndex(I::IntTuple{N}) where {N} = new{N, typeof(I)}(I)
end

HierarchicalIndex(I::GenIntTuple...) = HierarchicalIndex(I)

Base.show(io::IO, i::HierarchicalIndex) = (print(io, "HierarchicalIndex"); show(io, i.I))

Base.zero(::HierarchicalIndex{N,S}) where {N, S} = HierarchicalIndex(repeat_like(S, 0))
Base.zero(::Type{HierarchicalIndex{N, S}}) where {N, S} = HierarchicalIndex(repeat_like(S, 0))
Base.oneunit(::HierarchicalIndex{N, S}) where {N, S} = HierarchicalIndex(repeat_like(S, 1))
Base.oneunit(::Type{HierarchicalIndex{N, S}}) where {N, S} = HierarchicalIndex(repeat_like(S, 1))

Base.:(==)(a::HierarchicalIndex{N}, b::HierarchicalIndex{N}) where N = a.I == b.I
Base.:(<)(a::HierarchicalIndex{N}, b::HierarchicalIndex{N}) where N = a.I < b.I
Base.:(!=)(a::HierarchicalIndex{N}, b::HierarchicalIndex{N}) where N = a.I != b.I

struct HierarchicalIndices{N, S}
    shape::IntTuple{N}
    HierarchicalIndices(I::IntTuple{N}) where {N} = new{N, typeof(I)}(I)
end

HierarchicalIndices(I::GenIntTuple...) = HierarchicalIndices(I)
Base.show(io::IO, i::HierarchicalIndices) = (print(io, "HierarchicalIndices"); show(io, i.I))

Base.first(x::HierarchicalIndices) = HierarchicalIndex(repeat_like(x.shape, 1))
Base.last(x::HierarchicalIndices) = HierarchicalIndex(x.shape)
Base.length(x::HierarchicalIndices) = capacity(x.shape)
=#

HierarchicalIndices(x::MoYeArray) = Base.oneto(x.layout.shape)
