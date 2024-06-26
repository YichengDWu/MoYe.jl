@inline function axpby!(a::Number, X::MoYeArray{T}, b::Number, Y::MoYeArray{T}) where {T}
    x, y = ManualMemory.preserve_buffer(X), ManualMemory.preserve_buffer(Y)
    vx, vy = ViewEngine(pointer(X)), ViewEngine(pointer(Y))
    GC.@preserve x y begin
        @loopinfo unroll for i in eachindex(vx)
            @inbounds vy[i] = iszero(b) ? a * vx[i] : a * vx[i] + b * vy[i]
        end
    end
end

@inline gemm!(A::MoYeArray, B::MoYeArray, C::MoYeArray) = gemm!(C, A, B, C)
@inline gemm!(mma::AbstractMMAAtom, A::MoYeArray, B::MoYeArray, C::MoYeArray) = gemm!(mma,C,A,B,C)

function gemm!(D::MoYeArray{TD}, A::MoYeArray{TA}, B::MoYeArray{TB}, C::MoYeArray{TC}) where {TD,TA,TB,TC}
    @inline
    return gemm!(MMAAtom{UniversalFMA{TD,TA,TB,TC}}(), D, A, B, C)
end

# element-wise multiplication (1,1,1,1)
function gemm!(mma_atom::AbstractMMAAtom, D::LocalArray{DT,1},  A::LocalArray{DA,1},
               B::LocalArray{DB,1}, C::LocalArray{DC,1}) where {DT,DA,DB,DC}
    @inline
    apply(mma_atom, D, A, B, C)
    return nothing
end

# outer product (2,1,1,2) -> (2,2,2,2)
@generated function gemm!(mma_atom::AbstractMMAAtom, D::LocalArray{DT,2},  A::LocalArray{DA,1},
               B::LocalArray{DB,1}, C::LocalArray{DC,2}) where {DT,DA,DB,DC}
    @assert size(layout(A), _1) == size(layout(C), _1) == size(layout(D), _1) # M
    @assert size(layout(B), _1) == size(layout(C), _2) == size(layout(D), _2) # N
    return quote
        Base.@_inline_meta
        gemm!(mma_atom, D, append_dim(A, _2), append_dim(B, _2), C)
    end
end

# matrix multiplication (2,2,2,2) -> (3,3,3,3)
@generated function gemm!(mma_atom::AbstractMMAAtom, D::LocalArray{DT,2},  A::LocalArray{DA,2},
               B::LocalArray{DB,2}, C::LocalArray{DC,2}) where {DT,DA,DB,DC}
    @assert size(layout(A), _1) == size(layout(C), _1) == size(layout(D), _1) # M
    @assert size(layout(B), _1) == size(layout(C), _2) == size(layout(D), _2) # N
    @assert size(layout(A), _2) == size(layout(B), _2) # K
    return quote
        Base.@_inline_meta
        gemm!(mma_atom,
              prepend_dim(D, _3), prepend_dim(A, _3),
              prepend_dim(B, _3), prepend_dim(C, _3))
    end
end

# batched outer product (3,2,2,3) -> (1,1,1,1)
@generated function gemm!(mma_atom::AbstractMMAAtom, 
                          D::LocalArray{DT,3}, A::LocalArray{DA,2},
                          B::LocalArray{DB,2}, C::LocalArray{DC,3}) where {DT,DA,DB,DC}
    @assert size(layout(A), _2) == size(layout(C), _2) == size(layout(D), _2) # M
    @assert size(layout(B), _2) == size(layout(C), _3) == size(layout(D), _3) # N
    @assert size(layout(C), _1) == size(layout(D), _1)

    M = size(layout(A), _2)
    return quote
        Base.@_inline_meta
        @loopinfo unroll for n in axes(B, 2)
            @loopinfo unroll for m in axes(A, 2)
                ms = Bool(n & 1) ? m : $(M()+_1)-m
                gemm!(mma_atom, view(D, :, ms, n), view(A, :, ms), view(B, :, n), view(C, :, ms, n))
            end
        end
        return nothing
    end
end

# batched matrix multiplication (3,3,3,3) -> (3,2,2,3)
@generated function gemm!(mma_atom::AbstractMMAAtom, D::LocalArray{DT,3},  A::LocalArray{DA,3},
               B::LocalArray{DB,3}, C::LocalArray{DC,3}) where {DT,DA,DB,DC}
    @assert size(layout(A), _2) == size(layout(C), _2) == size(layout(D), _2) # M
    @assert size(layout(B), _2) == size(layout(C), _3) == size(layout(D), _3) # N
    @assert size(layout(A), _3) == size(layout(B), _3) # K
    @assert size(layout(C), _1) == size(layout(D), _1)

    return quote
        Base.@_inline_meta
        @loopinfo unroll for k in axes(A, 3)
            gemm!(mma_atom,  D, view(A, :, :, k), view(B, :, :, k), C)
        end
        return nothing
    end
end

# (2,2,2,2) -> (3,3,3,3)
@generated function gemm!(mma_atom::AbstractMMAAtom,
               D::LocalArray{DT, 2}, 
               A::SharedArray{DA, 2},
               B::SharedArray{DB, 2},
               C::LocalArray{DC, 2}) where {DT,DA,DB,DC}
    @assert size(layout(A), _1) == size(layout(C), _1) == size(layout(D), _1) # M
    @assert size(layout(B), _1) == size(layout(C), _2) == size(layout(D), _2) # N
    @assert size(layout(A), _2) == size(layout(B), _2) # K

    @assert size(layout_a(mma_atom()), _2) == One()
    @assert size(layout_b(mma_atom()), _2) == One()
    @assert size(layout_c(mma_atom()), _2) == One()

    return quote
        gemm!(mma_atom,
        prepend_dim(D,  _3), prepend_dim(A,  _3),
        prepend_dim(B,  _3), prepend_dim(C,  _3)) 
    end
end

# (3,3,3,3) -> (3,2,2,3)
@generated function gemm!(mma_atom::AbstractMMAAtom, D::LocalArray{DT,3},  A::SharedArray{DA,3},
               B::SharedArray{DB,3}, C::LocalArray{DC,3}) where {DT, DA, DB, DC}
    @assert size(layout(A), _2) == size(layout(C), _2) == size(layout(D), _2) # M
    @assert size(layout(B), _2) == size(layout(C), _3) == size(layout(D), _3) # N
    @assert size(layout(A), _3) == size(layout(B), _3) # K
    @assert size(layout(C), _1) == size(layout(D), _1)

    return quote
        Base.@_inline_meta
        rA = make_fragment_A(mma_atom, A)
        rB = make_fragment_B(mma_atom, B)

        @loopinfo unroll for k in axes(A,3)
            _copyto!(view(rA, :, :, k), view(A, :, :, k))
            _copyto!(view(rB, :, :, k), view(B, :, :, k))
            gemm!(mma_atom, D, view(rA, :, :, k), view(rB, :, :, k), C) 
        end
        return nothing
    end
end

function gemm!(thr_mma::ThrMMA, alpha, A::SharedArray{TA,2}, B::SharedArray{TB,2},
               beta, C::SharedArray{TC,2}, transform_A, transform_B) where {TA,TB,TC}
    @assert size(layout(A), 1) == size(layout(C), 1) # M
    @assert size(layout(B), 1) == size(layout(C), 2) # N
    @assert size(layout(A), 2) == size(layout(B), 2) # K

    @assert Core.Compiler.return_type(transform_A, Tuple{TA}) == TA
    @assert Core.Compiler.return_type(transform_B, Tuple{TB}) == TB

    M = size(layout(C), 1)
    N = size(layout(C), 2)
    K = size(layout(A), 2)

    BLK_M = tile_size(thr_mma, 1)
    BLK_N = tile_size(thr_mma, 2)
    BLK_K = tile_size(thr_mma, 3)

    m_residue = M - BLK_M * (cld(M, BLK_M) - One())
    n_residue = N - BLK_N * (cld(N, BLK_N) - One())
    k_residue = K - BLK_K * cld(K, BLK_K)

    sA = MoYeArray(pointer(A, (1, k_residue+1)), layout(A))
    sB = MoYeArray(pointer(B, (1, k_residue+1)), layout(B))

    rounded_sA = sA ∘ (cld(M, BLK_M) * BLK_M, cld(K, BLK_K) * BLK_K)
    rounded_sB = sB ∘ (cld(N, BLK_N) * BLK_N, cld(K, BLK_K) * BLK_K)
    rounded_sC = sC ∘ (cld(M, BLK_M) * BLK_M, cld(N, BLK_N) * BLK_N)

    thr_A = partition_A(thr_mma, rounded_sA)
    thr_B = partition_B(thr_mma, rounded_sB)
    thr_C = partition_C(thr_mma, rounded_sC)

    rA = make_fragment_A(thr_mma, thr_A)
    rB = make_fragment_B(thr_mma, thr_B)
    rC = make_fragment_C(thr_mma, thr_C)

    # predication
    thr_pA = MoYeArray{Bool}(undef, static_size(thr_A, 1))
    thr_pB = MoYeArray{Bool}(undef, static_size(thr_B, 1))

    cA = make_identity_array((BLK_M, BLK_K))
    cB = make_identity_array((BLK_N, BLK_K))

    thr_cA = partition_A(thr_mma, cA)
    thr_cB = partition_B(thr_mma, cB)

    @loopinfo unroll for i in size(layout(thr_pA))
        @inbounds thr_pA[i] = elem_less(thr_cA[i][1], m_residue)
    end

    @loopinfo unroll for i in size(layout(thr_pB))
        @inbounds thr_pB[i] = elem_less(thr_cB[i][1], n_residue)
    end

    # load A
    @loopinfo unroll for i in axes(thr_A, 1)
        if k_residue == Zero() || thr_cA[i][2] > -k_residue
            @loopinfo unroll for m in axes(thr_A, 2)
                @inbounds rA[i, m, 1] = (m_residue == BLK_M || m <= static_size(thr_A, 2) || thr_pA[i]) ?
                                            transform_A(thr_A[i, m, 1]) : zero(TA)
            end
        end
    end

    # load B
    @loopinfo unroll for i in axes(thr_B, 1)
        if k_residue == Zero() || thr_cB[i][2] > -k_residue
            @loopinfo unroll for n in axes(thr_B, 2)
                @inbounds rB[i, n, 1] = (n_residue == BLK_N || n <= static_size(thr_B, 2) || thr_pB[i]) ?
                                            transform_B(thr_B[i, n, 1]) : zero(TB)
            end
        end
    end

    zeros!(rC)

    K_BLK_MAX = static_size(thr_A, 3)

    @loopinfo unroll for k_blk in 1:K_BLK_MAX
        if k_blk < K_BLK_MAX
            k_next = k_blk + 1
            @loopinfo unroll for m in axes(thr_A, 2)
                @loopinfo unroll for i in axes(thr_A, 1)
                    rA[i, m, k_next] = (m_residue == BLK_M || m <= static_size(thr_A, 2) || thr_pA[i]) ?
                                            transform_A(thr_A[i, m, k_next]) : zero(TA)
                end
            end

            @loopinfo unroll for n in axes(thr_B, 2)
                @loopinfo unroll for i in axes(thr_B, 1)
                    rB[i, n, k_next] = (n_residue == BLK_N || n <= static_size(thr_B, 2) || thr_pB[i]) ?
                                            transform_B(thr_B[i, n, k_next]) : zero(TB)
                end
            end

            gemm!(thr_mma, view(rA, :, :, k_blk), view(rB, :, :, k_blk), rC)
        end

    end

    ############
    # Epilogue #
    ############

    cC = make_identity_array((BLK_M, BLK_N))
    thr_cC = partition_C(thr_mma, cC)

    is_beta_zero = iszero(beta)

    @loopinfo unroll for m in axes(thr_C, 2)
        @loopinfo unroll for n in axes(thr_C, 3)
            @loopinfo unroll for i in axes(thr_C, 1)
                if (m_residue == BLK_M || m <= static_size(thr_C, 2) || thr_cC[i][1] <= m_residue) &&
                   (n_residue == BLK_N || n <= static_size(thr_C, 3) || thr_cC[i][2] <= n_residue)
                    @inbounds thr_C[i, m, n] = is_beta_zero ? alpha * rC[i, m, n] : alpha * rC[i, m, n] + beta * thr_C[i, m, n]
                end
            end
        end
    end
    return nothing
end

function gemm!(thr_mma::ThrMMA, alpha, A::SharedArray{TA,2}, B::SharedArray{TB,2},
               beta, C::SharedArray{TC,2}) where {TA,TB,TC}
    gemm!(thr_mma, alpha, A, B, beta, C, identity, identity)
end
