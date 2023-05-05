@inline function axpby!(a::Number, X::MoYeArray{T}, b::Number, Y::MoYeArray{T}) where {T}
    x, y = ManualMemory.preserve_buffer(X), ManualMemory.preserve_buffer(Y)
    vx, vy = ViewEngine(engine(X)), ViewEngine(engine(Y))
    GC.@preserve x y begin
        @loopinfo unroll  for i in eachindex(vx)
            @inbounds vy[i] = iszero(b) ? a * vx[i] : a * vx[i] + b * vy[i]
        end
    end
end

# used for dispatching
const LocalArray{T,N} = MoYeArray{T,N, ViewEngine{T, Ptr{T}}}

gemm!(A::MoYeArray, B::MoYeArray, C::MoYeArray) = gemm!(C,A,B,C)
#gemm!(mma::MMAAtom, A::MoYeArray, B::MoYeArray, C::MoYeArray) = gemm!(mma,C,A,B,C)

function gemm!(D::MoYeArray{TD}, A::MoYeArray{TA}, B::MoYeArray{TB}, C::MoYeArray{TC}) where {TD,TA,TB,TC}
    mma_atom = MMAAtom{UniversalFMA{TD,TA,TB,TC}}()
    gemm!(mma_atom, A, B, C, D)
end

# element-wise multiplication
function gemm!(mma_atom::MMAAtom, D::LocalArray{DT,1},  A::LocalArray{DA,1},
               B::LocalArray{DB,1}, C::LocalArray{DC,1}) where {DT,DA,DB,DC}
    apply(mma_atom, D, A, B, C)
end

# outer product
function gemm!(mma_atom::MMAAtom, D::LocalArray{DT,2},  A::LocalArray{DA,1},
               B::LocalArray{DB,1}, C::LocalArray{DC,2}) where {DT,DA,DB,DC}

    gemm(mma_atom, D, unsqueeze(A, StaticInt{2}()), unsqueeze(B, StaticInt{2}()), C)
end
