#=
abstract type EBO{N, T, IsEmpty} end

struct EmptyEBO{N,T} <: EBO{N, T, true}
    EmptyEBO{N,T}() where {N, T} = new{N::UInt, T}()
    EmptyEBO{N,T}(@nospecialize ::T) where {N, T} = new{N,T}()
end

struct NonEmptyEBO{N, T} <: EBO{N, T, false}
    t_::T
    NonEmptyEBO{N,T}(t_::T) where {N, T} = new{N::UInt, T}(t_)
    NonEmptyEBO{N,T}(t_::U) where {N, T, U} = new{N, T}(convert(T, t_))
end

EBO{N, T, true}() where {N, T} = EmptyEBO{N, T}()

EBO{N, T, true}(u::T) where {N, T} = EmptyEBO{N, T}(u)

getv(::EBO{N,T,true}) where {N, T} = T()

EBO{N, T, false}() where {N, T} = NonEmptyEBO{N, T}(zero(T))

EBO{N, T, false}(u) where {N, T} = NonEmptyEBO{N, T}(u)

getv(x::EBO{N,T,false}) where {N, T} = x.t_

const index_sequence = NTuple{N,UInt}

TupleBase = NTuple{N, EBO} where N

function make_index_sequence(N, M...)
    iszero(N) ? M : make_index_sequence(N-1, N-1, M...)
end
=#

istuple = Base.Fix2(isa, Tuple)
