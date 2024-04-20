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

function Base.show(io::IO, ::MIME"text/plain",
    @nospecialize(x::StaticInt))
    print(io, "_" * repr(known(typeof(x))))
    nothing
end


@generated function Base.abs(::StaticInt{N}) where {N}
    return quote
        Base.@_inline_meta
        return $(StaticInt{abs(N)}())
    end
end

const _0 = StaticInt{0}()
const _1 = StaticInt{1}()
const _2 = StaticInt{2}()
const _3 = StaticInt{3}()
const _4 = StaticInt{4}()
const _5 = StaticInt{5}()
const _6 = StaticInt{6}()
const _7 = StaticInt{7}()
const _8 = StaticInt{8}()
const _9 = StaticInt{9}()
const _10 = StaticInt{10}()
const _16 = StaticInt{16}()
const _32 = StaticInt{32}()
const _64 = StaticInt{64}()
const _128 = StaticInt{128}()
const _256 = StaticInt{256}()
const _512 = StaticInt{512}()
const _1024 = StaticInt{1024}()
const _2048 = StaticInt{2048}()
const _4096 = StaticInt{4096}()
const _8192 = StaticInt{8192}()