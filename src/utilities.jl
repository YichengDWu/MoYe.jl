uint_bit(::StaticInt{8}) = UInt8
uint_bit(::StaticInt{16}) = UInt16
uint_bit(::StaticInt{32}) = UInt32
uint_bit(::StaticInt{64}) = UInt64
uint_bit(::StaticInt{128}) = UInt128

uint_bytes(::StaticInt{N}) where {N} = uint_bit(static(8*N))

@generated sizeof_bits(::Type{T}) where {T} = :($(static(sizeof(T)*8)))

@inline Base.:(==)(::StaticInt{N}, ::StaticInt{N}) where {N} = true
@inline Base.:(==)(@nospecialize(x::StaticInt), @nospecialize(y::StaticInt)) = false

@inline Base.:(*)(::StaticInt{1}, x::Int) = x
@inline Base.:(*)(x::Int, ::StaticInt{1}) = x
@inline Base.:(*)(::StaticInt{0}, ::Int) = Zero()
@inline Base.:(*)(::Int, ::StaticInt{0}) = Zero()

@inline Base.:(+)(::StaticInt{0}, x::Int) = x
@inline Base.:(+)(x::Int, ::StaticInt{0}) = x

@inline @generated Base.abs(::StaticInt{N}) where {N} = :($(StaticInt{abs(N)}()))
