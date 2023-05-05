uint_bit(::StaticInt{8}) = UInt8
uint_bit(::StaticInt{16}) = UInt16
uint_bit(::StaticInt{32}) = UInt32
uint_bit(::StaticInt{64}) = UInt64
uint_bit(::StaticInt{128}) = UInt128

uint_bytes(::StaticInt{N}) where {N} = uint_bit(static(8*N))

@traitdef IsEqual{X,Y}
@traitimpl IsEqual{X,Y} <- isequal(X,Y)
isequal(::Type{T}, ::Type{S}) where {T,S} = T == S

@traitdef IsGreater{X,Y}
@traitimpl IsGreater{X,Y} <- isgreater(X,Y)
isgreater(::Type{StaticInt{N}}, ::Type{StaticInt{M}}) where {N,M} = N>M

@traitdef IsLess{X,Y}
@traitimpl IsLess{X,Y} <- isless(X,Y)
isless(::Type{StaticInt{N}}, ::Type{StaticInt{M}}) where {N,M} = N<M

@traitdef IsSizeEqual{X,Y}
@traitimpl IsSizeEqual{X,Y} <- issizeequal(X,Y)
issizeequal(::Type{T}, ::Type{S}) where {T,S} = sizeof(T) == sizeof(S)

@traitdef IsSizeLess{X,Y}
@traitimpl IsSizeLess{X,Y} <- issizeless(X,Y)
issizeless(::Type{T}, ::Type{S}) where {T,S} = sizeof(T) < sizeof(S)

@traitdef IsSizeGreater{X,Y}
@traitimpl IsSizeGreater{X,Y} <- issizegreater(X,Y)
issizegreater(::Type{T}, ::Type{S}) where {T,S} = sizeof(T) > sizeof(S)

@traitdef HasColon{X}
@traitimpl HasColon{X} <- hascolon(X)
@generated hascolon(::Type{T}) where T = :($(Colon ∈ T.parameters))

@traitdef IsDivisible{X,Y}
@traitimpl IsDivisible{X,Y} <- isdivisible(X,Y)
@generated isdivisible(::Type{StaticInt{N}}, ::Type{StaticInt{M}}) where {N,M} = :($(N%M == 0))
