uint_bit(::StaticInt{8}) = UInt8
uint_bit(::StaticInt{16}) = UInt16
uint_bit(::StaticInt{32}) = UInt32
uint_bit(::StaticInt{64}) = UInt64
uint_bit(::StaticInt{128}) = UInt128

uint_bytes(::StaticInt{N}) where {N} = uint_bit(static(8*N))

@generated sizeof_bits(::Type{T}) where {T} = :($(static(sizeof(T)*8)))

@inline Base.:(==)(::StaticInt{N}, ::StaticInt{N}) where {N} = true
@inline Base.:(==)(@nospecialize(x::StaticInt), @nospecialize(y::StaticInt)) = false

@inline Base.:(*)(::Type{StaticInt{N}}, ::Type{StaticInt{M}}) where {N, M} = StaticInt{N*M}

@generated function Base.abs(::StaticInt{N}) where {N}
    return quote
        Base.@_inline_meta
        return $(StaticInt{abs(N)}())
    end
end

const Two = StaticInt{2}
const Three = StaticInt{3}

const 𝟎 = Zero()
const 𝟏 = One()
const 𝟐 = Two()
const 𝟑 = Three()
