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
const SharedArray{T,N} = MoYeArray{T,N, ViewEngine{T, LLVMPtr{T, AS.Shared}}}

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

    gemm(mma_atom, D, append_dim(A, StaticInt{2}()), append_dim(B, StaticInt{2}()), C)
end

# matrix multiplication
function gemm!(mma_atom::MMAAtom, D::LocalArray{DT,2},  A::LocalArray{DA,2},
               B::LocalArray{DB,2}, C::LocalArray{DC,2}) where {DT,DA,DB,DC}
    gemm(mma_atom,
         prepend(D,  StaticInt{3}()), prepend(A,  StaticInt{3}()),
         prepend(B,  StaticInt{3}()), prepend(C,  StaticInt{3}()))
end

# batched outer product
function gemm!(mma_atom::MMAAtom, D::LocalArray{DT,3},  A::LocalArray{DA,2},
               B::LocalArray{DB,2}, C::LocalArray{DC,3}) where {DT,DA,DB,DC}
    M = size(A, 2)
    N = size(B, 2)
    @loopinfo unroll for n in 1:N
        @loopinfo unroll for m in 1:M
            ms = (n & 1) ? m : M+1-m
            gemm(mma_atom, view(D, :, ms, n), view(A, :, ms), view(B, :, n), view(C, :, ms, n))
        end
    end
end

# batched matrix multiplication
function gemm!(mma_atom::MMAAtom, D::LocalArray{DT,3},  A::LocalArray{DA,3},
               B::LocalArray{DB,3}, C::LocalArray{DC,3}) where {DT,DA,DB,DC}
    @loopinfo unroll for k in axes(A, 3)
        gemm!(mma_atom,  D, view(A, :, :, k), view(B, :, :, k), C)
    end
end

function gemm!()

end
