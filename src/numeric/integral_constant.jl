#=struct Constant{T, v}
    value::T
    Constant{T, v}(value::T) where {T, v} = new{T, v::T}(value)
end

Constant{T}(::Val{v}) where {T, v} = Constant{T, v}(T(v))
Constant(::Val{v}) where {v} = Constant{typeof(v)}(v)

Base.convert(::Type{T}, @nospecialize(c::Constant)) where T = convert(T, c.value)

(c::Constant{T, v})() where {T, v} = c.value

Base.eltype(::Type{<:Constant{T}}) where T = T
Base.:(==)(@nospecialize(c1::Constant), @nospecialize(c2::Constant)) = c1() == c2()
Base.isinteger(@nospecialize x::StaticInteger) = true

const BoolConstant{bool} = Constant{Bool, bool}

True = BoolConstant{true}
False = BoolConstant{false}
=#
