# Here I use a different approach from copy atoms, just to see if it simplifies things.
# aka use parametric types instead of subtyping
abstract type AbstractMMAAtom{Traits} end

mma_traits(mma_atom::AbstractMMAAtom) = mma_atom.traits

function apply(mma_atom::AbstractMMAAtom, D::MoYeArray{TD, 1}, A::MoYeArray{TA, 1},
               B::MoYeArray{TB, 1}, C::MoYeArray{TC, 1}) where {TD, TA, TB, TC}
    @inline
    return mma_unpack!(mma_traits(mma_atom), D, A, B, C)
end
function apply(mma_atom::AbstractMMAAtom, A::MoYeArray, B::MoYeArray, C::MoYeArray)
    @inline
    return apply(mma_traits(mma_atom), C, A, B, C)
end

function make_fragment_C(m::AbstractMMAAtom, C::MoYeArray{T, N}) where {T, N}
    @assert N ≥ 3
    @assert size(layout(C), 1) == size(mma_traits(m).Clayout, 2)
    return MoYeArray{frgtype_c(mma_traits(m))}(undef, shape(C)) # (V, M, N)
end

# Hopper needs to specialize on make_fragment_A and make_fragment_B
function make_fragment_A(m::AbstractMMAAtom, A::MoYeArray{T, N}) where {T, N}
    @assert N ≥ 3
    @assert size(layout(A), 1) == size(mma_traits(m).Alayout, 2)
    return make_fragment_like(frgtype_a(mma_traits(m)), A) # (V, M, K)
end

function make_fragment_B(m::AbstractMMAAtom, B::MoYeArray{T, N}) where {T, N}
    @assert N ≥ 3
    @assert size(layout(B), 1) == size(mma_traits(m).Blayout, 2)
    return make_fragment_like(frgtype_b(mma_traits(m)), B) # (V, K, N)
end

struct MMAAtom{Traits} <: AbstractMMAAtom{Traits}
    traits::Traits
    function MMAAtom{Traits}(args...) where {Traits <: MMATraits}
        traits = Traits(args...)
        return new{typeof(traits)}(traits)
    end
    function MMAAtom{OP}(args...) where {OP <: AbstractMMAOP}
        traits = MMATraits{OP}(args...)
        return new{typeof(traits)}(traits)
    end
end

function Base.show(io::IO, m::MMAAtom)
    println(io, "MMAAtom")
    println(io, "  Thread ID: ", mma_traits(m).threadid)
    println(io, "  Layout_A_TV: ", mma_traits(m).Alayout)
    println(io, "  Layout_B_TV: ", mma_traits(m).Blayout)
    return println(io, "  Layout_C_TV: ", mma_traits(m).Clayout)
end


struct TiledMMA{Atom, TiledThr, TiledVal, TiledPerm, TM, TL, TIL}
    mma_atom::Atom
    atom_layout_MNK::TiledThr
    val_layout_MNK::TiledVal
    permutations_MNK::TiledPerm
    tiled_MNK::TM
    thr_layout_VMNK::TL
    tid_layout::TIL
end

function TiledMMA(atom::MMAAtom,
                  atom_layout_MNK::Layout{3}=@Layout((1, 1, 1)),
                  val_layout_MNK::Layout{3}=@Layout((1, 1, 1)),
                  permutations_MNK::Tile{3}=(:, :, :))
    MNK = atom.traits.MNK
    threadid = mma_traits(atom).threadid

    tiled_MNK = (
        MNK[1] * size(atom_layout_MNK, 1) * size(val_layout_MNK, 1),
        MNK[2] * size(atom_layout_MNK, 2) * size(val_layout_MNK, 2),
        MNK[3] * size(atom_layout_MNK, 3) * size(val_layout_MNK, 3)
    )
    thr_layout_VMNK = tiled_product(threadid, atom_layout_MNK)
    tid_layout = right_inverse(thr_layout_VMNK)
    return TiledMMA{typeof(mma_atom), typeof(atom_layout_MNK), typeof(val_layout_MNK), typeof(permutations_MNK),
                    typeof(tiled_MNK), typeof(thr_layout_VMNK), typeof(tid_layout)}(atom,
                    atom_layout_MNK, val_layout_MNK, permutations_MNK, tiled_MNK, thr_layout_VMNK, tid_layout)
end

function make_tiled_mma(mma_atom::AbstractMMAAtom, thr_layout::Layout=@Layout((1, 1, 1)),
                        val_layout::Layout=@Layout((1, 1, 1)), permutations::Tile=(:, :, :))
    thr_layout_mnk = append(thr_layout, @Layout(1, 0), static(3))
    val_layout_mnk = append(val_layout, @Layout(1, 0), static(3))
    permutation_mnk = append(permutations, :, static(3))

    return TiledMMA(mma_atom, thr_layout_mnk, val_layout_mnk, permutation_mnk)
end

function make_tiled_mma(::OP, thr_layout::Layout=@Layout((1, 1, 1)),
                        val_layout::Layout=@Layout((1, 1, 1)), permutations::Tile=(:, :, :)) where {OP <: AbstractMMAOP}
    return make_tiled_mma(MMAAtom{OP}(), thr_layout, val_layout, permutations)
end

function thrfrg_C(m::TiledMMA, C::MoYeArray{T,2}) where {T}
    tiled_MNK = m.tiled_MNK
    permutations_MNK = m.permutations_MNK
    thr_layout_VMNK = m.thr_layout_VMNK
    atom_MNK = m.mma_atom.traits.MNK

    @assert size(layout(C), 1) % size(tiled_MNK, 1) == Zero()
    @assert size(layout(C), 2) % size(tiled_MNK, 2) == Zero()
    t_tile = (left_inverse(permutations_MNK[1]), left_inverse(permutations_MNK[2]))
    t_array = logical_divide(C, t_tile)

    a_tile = (make_layout(atom_MNK[1]), make_layout(atom_MNK[3]))
    a_array = zipped_divide(t_array, a_tile)

    tv_array = compose(a_array, m.mma_atom.traits.Clayout, :)

    thr_tile = (:, (make_layout(thr_layout_VMNK[2]), make_layout(thr_layout_VMNK[3])))
    thr_array = zipped_divide(tv_array, thr_tile)
    return thr_array
end

function tidfrg_C(m::TiledMMA, C::MoYeArray)
    return compose(thrfrg_C(m, C), m.tid_layout, :)
end

function thrfrg_A(m::TiledMMA, A::MoYeArray{T,2}) where {T}
    tiled_MNK = m.tiled_MNK
    permutations_MNK = m.permutations_MNK
    thr_layout_VMNK = m.thr_layout_VMNK
    atom_MNK = m.mma_atom.traits.MNK

    @assert size(layout(A), 1) % size(tiled_MNK, 1) == Zero()
    @assert size(layout(A), 2) % size(tiled_MNK, 3) == Zero()

    t_tile = (left_inverse(permutations_MNK[1]), left_inverse(permutations_MNK[3]))
    t_array = logical_divide(A, t_tile)

    a_tile = (make_layout(atom_MNK[1]), make_layout(atom_MNK[3]))
    a_array = zipped_divide(t_array, a_tile)

    tv_array = compose(a_array, m.mma_atom.traits.Alayout, :)
    thr_tile = (:, (make_layout(thr_layout_VMNK[2]), make_layout(thr_layout_VMNK[4])))
    thr_array = zipped_divide(tv_array, thr_tile)
    return thr_array
end

function tidfrg_A(m::TiledMMA, A::MoYeArray)
    thr_layout_VMNK = m.thr_layout_VMNK
    atile = (:, (make_layout((thr_layout_VMNK[2], thr_layout_VMNK[3]), (One(), Zero())), :))
    return compose(compose(thrfrg_A(m, A), atile, :), m.tid_layout, :)
end

function thrfrg_B(m::TiledMMA, B::MoYeArray{T, 2}) where {T}
    tiled_MNK = m.tiled_MNK
    permutations_MNK = m.permutations_MNK
    thr_layout_VMNK = m.thr_layout_VMNK
    atom_MNK = m.mma_atom.traits.MNK

    @assert size(layout(B), 1) % size(tiled_MNK, 2) == Zero()
    @assert size(layout(B), 2) % size(tiled_MNK, 3) == Zero()

    t_tile = (left_inverse(permutations_MNK[2]), left_inverse(permutations_MNK[3]))
    t_array = logical_divide(B, t_tile)

    a_tile = (make_layout(atom_MNK[2]), make_layout(atom_MNK[3]))
    a_array = zipped_divide(t_array, a_tile)

    tv_array = compose(a_array, m.mma_atom.traits.Blayout, :)
    thr_tile = (:, (make_layout(thr_layout_VMNK[3]), make_layout(thr_layout_VMNK[4])))
    thr_array = zipped_divide(tv_array, thr_tile)
    return thr_array
end

function tidfrg_B(m::TiledMMA, B::MoYeArray)
    thr_layout_VMNK = m.thr_layout_VMNK
    btile = (:, (make_layout((thr_layout_VMNK[2], thr_layout_VMNK[3]), (One(), Zero())), :))
    return compose(compose(thrfrg_B(m, B), btile, :), m.tid_layout, :)
end

function Base.show(io::IO, m::TiledMMA{Atom, TiledThr, TiledVal, TiledPerm}) where {Atom, TiledThr, TiledVal, TiledPerm}
    println(io, "TiledMMA")
    println(io, "  Tiled Thr: ", TiledThr())
    println(io, "  Tiled Val: ", TiledVal())
    println(io, "  Tiled Perm: ", TiledPerm())
    println(io, "  Tiled MNK: ", m.tiled_MNK)
    println(io, "  Thr Layout VMNK: ", m.thr_layout_VMNK)
    return show(io, m.mma_atom)
end

struct ThrMMA{TA<:TiledMMA, ThrVMNK}
    tiled_mma::TA
    thr_vmnk::ThrVMNK
end

function get_slice(m::TiledMMA, thr_idx::Int)
    thr_vmnk = get_congr_coord(m.thr_layout_VMNK, thr_idx)
    return ThrMMA(m, thr_vmnk)
end

function get_thread_slice(m::TiledMMA, thr_idx::Int)
    @inline
    return get_slice(m, thr_idx)
end

function partition_C(m::ThrMMA, C::MoYeArray)
    thr_array = MoYeArray(pointer(C), thrfrg_C(m.tiled_mma, layout(C)))
    thr_vmn = (m.thr_vmnk[1], (m.thr_vmnk[2], m.thr_vmnk[3]))
    return view(thr_array, thr_vmn, (:, repeat(:, rank(shape(layout(thr_array)[1][1])))))
end

function partition_A(m::ThrMMA, A::MoYeArray)
    thr_array = MoYeArray(pointer(A), thrfrg_A(m.tiled_mma, layout(A)))
    thr_vmk = (m.thr_vmnk[1], (m.thr_vmnk[2], m.thr_vmnk[4]))
    return view(thr_array, thr_vmk, (:, repeat(:, rank(shape(layout(thr_array)[1][1])))))
end

function partition_B(m::ThrMMA, B::MoYeArray)
    thr_array = MoYeArray(pointer(B), thrfrg_B(m.tiled_mma, layout(B)))
    thr_vnk = (m.thr_vmnk[1], (m.thr_vmnk[3], m.thr_vmnk[4]))
    return view(thr_array, thr_vnk, (:, repeat(:, rank(shape(layout(thr_array)[1][1])))))
end

function partiton_fragment_C(m::ThrMMA, C::MoYeArray)
    return make_fragment_C(m.mma_atom, partition_C(m, C))
end

function partiton_fragment_A(m::ThrMMA, A::MoYeArray)
    return make_fragment_A(m.mma_atom, partition_A(m, A))
end

function partiton_fragment_B(m::ThrMMA, B::MoYeArray)
    return make_fragment_B(m.mma_atom, partition_B(m, B))
end

function Base.show(io::IO, m::ThrMMA{TA, ThrVMNK}) where {TA, ThrVMNK}
    println(io, "ThrMMA")
    println(io, "  Thr VMNK: ", m.thr_vmnk)
    return show(io, m.tiled_mma)
end
