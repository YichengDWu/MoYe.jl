#=struct StaticUInt{N} <: Static.StaticInteger{N}
    StaticUInt{N}() where {N} = new{N::UInt}()
    StaticUInt(N::UInt) = new{N}()
    StaticUInt(@nospecialize N::StaticUInt) = N
    StaticUInt(::Val{N}) where {N} = StaticInt(N)
end

#const UIntType = Union{StaticUInt, UInt}
#UIntType(x::Integer) = UInt(x)
#UIntType(@nospecialize x::UIntType) = x

Base.zero(@nospecialize(::StaticUInt)) = StaticUInt{0x0000000000000000}()
Base.zero(@nospecialize T::Type{<:StaticUInt}) = StaticUInt(0x0000000000000000)

Base.one(@nospecialize(::StaticUInt)) = StaticUInt{0x0000000000000001}()
Base.one(@nospecialize T::Type{<:StaticUInt}) = StaticUInt(0x0000000000000001)

Static.static(x::UInt) = StaticUInt(x)
Static.dynamic(::StaticUInt{N}) where {N} = N

=#
Base.isnothing(@nospecialize x::Static.StaticNumber) = false
Base.cld(x::StaticInt, y::StaticInt) = div(x, y, RoundUp)
