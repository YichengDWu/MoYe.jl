@inline function axpby!(a::Number, X::MoYeArray{T}, b::Number, Y::MoYeArray{T}) where {T}
    x, y = ManualMemory.preserve_buffer(X), ManualMemory.preserve_buffer(Y)
    vx, vy = ViewEngine(engine(X)), ViewEngine(engine(Y))
    GC.@preserve x y begin
        @loopinfo unroll  for i in eachindex(vx)
            @inbounds vy[i] = iszero(b) ? a * vx[i] : a * vx[i] + b * vy[i]
        end
    end
end

gemm!(A::MoYeArray, B::MoYeArray, C::MoYeArray) = gemm!(C,A,B,C)
gemm!(mma::MMAAtom, A::MoYeArray, B::MoYeArray, C::MoYeArray) = gemm!(mma,C,A,B,C)

function gemm!(D::MoYeArray{TD}, A::MoYeArray{TA}, B::MoYeArray{TB}, C::MoYeArray{TC}) where {TD,TA,TB,TC}
    mma_atom = MMAAtom{UniversalFMA{TD,TA,TB,TC}}()
    gemm!(mma_atom, A, B, C, D)
end

# element-wise multiplication (1,1,1,1)
function gemm!(mma_atom::AbstractMMAAtom, D::LocalArray{DT,1},  A::LocalArray{DA,1},
               B::LocalArray{DB,1}, C::LocalArray{DC,1}) where {DT,DA,DB,DC}
    mma_unpack!(mma_atom, D, A, B, C)
end

# outer product (2,1,1,2)
function gemm!(mma_atom::AbstractMMAAtom, D::LocalArray{DT,2},  A::LocalArray{DA,1},
               B::LocalArray{DB,1}, C::LocalArray{DC,2}) where {DT,DA,DB,DC}
    @assert size(A.layout, 1) == size(C.layout, 1) == size(D.layout, 1) # M
    @assert size(B.layout, 1) == size(C.layout, 2 )== size(D.layout, 2) # N
    gemm(mma_atom, D, append_dim(A, StaticInt{2}()), append_dim(B, StaticInt{2}()), C)
end

# matrix multiplication (2,2,2,2)
function gemm!(mma_atom::AbstractMMAAtom, D::LocalArray{DT,2},  A::LocalArray{DA,2},
               B::LocalArray{DB,2}, C::LocalArray{DC,2}) where {DT,DA,DB,DC}
    @assert size(A.layout, 1) == size(C.layout, 1) == size(D.layout, 1) # M
    @assert size(B.layout, 1) == size(C.layout, 2) == size(D.layout, 2) # N
    @assert size(A.layout, 2) == size(B.layout, 2) # K

    gemm(mma_atom,
         prepend_dim(D,  StaticInt{3}()), prepend_dim(A,  StaticInt{3}()),
         prepend_dim(B,  StaticInt{3}()), prepend_dim(C,  StaticInt{3}()))
end

# batched outer product (3,2,2,3)
function gemm!(mma_atom::AbstractMMAAtom, D::LocalArray{DT,3},  A::LocalArray{DA,2},
               B::LocalArray{DB,2}, C::LocalArray{DC,3}) where {DT,DA,DB,DC}
    @assert size(A.layout, 2) == size(C.layout, 2) == size(D.layout, 2) # M
    @assert size(B.layout, 2) == size(C.layout, 3) == size(D.layout, 3) # N
    @assert size(A.layout, 1) == size(B.layout, 1) == size(C.layout, 1) == size(D.layout, 1) # V

    M = size(A, 2)
    N = size(B, 2)
    @loopinfo unroll for n in 1:N
        @loopinfo unroll for m in 1:M
            ms = (n & 1) ? m : M+1-m
            gemm(mma_atom, view(D, :, ms, n), view(A, :, ms), view(B, :, n), view(C, :, ms, n))
        end
    end
end

# batched matrix multiplication (3,3,3,3)
function gemm!(mma_atom::AbstractMMAAtom, D::LocalArray{DT,3},  A::LocalArray{DA,3},
               B::LocalArray{DB,3}, C::LocalArray{DC,3}) where {DT,DA,DB,DC}
    @assert size(A.layout, 2) == size(C.layout, 2) == size(D.layout, 2) # M
    @assert size(B.layout, 2) == size(C.layout, 3) == size(D.layout, 3) # N
    @assert size(A.layout, 3) == size(B.layout, 3) # K
    @assert size(A.layout, 1) == size(B.layout, 1) == size(C.layout, 1) == size(D.layout, 1) # V

    @loopinfo unroll for k in axes(A, 3)
        gemm!(mma_atom,  D, view(A, :, :, k), view(B, :, :, k), C)
    end
end

function gemm!(mma_atom::AbstractMMAAtom, D::LocalArray{DT,2},  A::SharedArray{DA,2},
               B::SharedArray{DB,2}, C::LocalArray{DC,2}) where {DT,DA,DB,DC}
    @assert size(A.layout, 1) == size(C.layout, 1) == size(D.layout, 1) # M
    @assert size(B.layout, 1) == size(C.layout, 2) == size(D.layout, 2) # N
    @assert size(A.layout, 2) == size(B.layout, 2) # K

    @assert size(mma_atom.traits.Alayout, 2) == One()
    @assert size(mma_atom.traits.Blayout, 2) == One()
    @assert size(mma_atom.traits.Clayout, 2) == One()

    gemm(mma_atom,
         prepend_dim(D,  StaticInt{3}()), prepend_dim(A,  StaticInt{3}()),
         prepend_dim(B,  StaticInt{3}()), prepend_dim(C,  StaticInt{3}()))
end

function gemm!(mma_atom::AbstractMMAAtom, D::LocalArray{DT,3},  A::SharedArray{DA,3},
               B::SharedArray{DB,3}, C::LocalArray{DC,3}) where {DT,DA,DB,DC}
    @assert size(A.layout, 2) == size(C.layout, 2) == size(D.layout, 2) # M
    @assert size(B.layout, 2) == size(C.layout, 3) == size(D.layout, 3) # N
    @assert size(A.layout, 3) == size(B.layout, 3) # K
    @assert size(A.layout, 1) == size(B.layout, 1) == size(C.layout, 1) == size(D.layout, 1) # V

    rA = make_fragment_A(mma_atom, A)
    rB = make_fragment_B(mma_atom, B)

    @loopinfo unroll for k in axes(A,3)
        cucopyto!(view(rA, :, :, k), view(A, :, :, k))
        cucopyto!(view(rB, :, :, k), view(B, :, :, k))
        gemm!(mma_atom, D, view(rA, :, :, k), view(rB, :, :, k), C)  # (3,2,2,3)
    end
end
