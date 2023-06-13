abstract type AbstractMMAAtom end

function make_fragment_C(m::AbstractMMAAtom, C::MoYeArray{T, N}) where {T, N}
    @assert N ≥ 3
    @assert size(layout(C), 1) == size(get_mma_traits(m).Clayout, 2)
    return MoYeArray{frgtype_c(get_mma_traits(m))}(undef, shape(C)) # (V, M, N)
end

# Hopper needs to specialize on make_fragment_A and make_fragment_B
function make_fragment_A(m::AbstractMMAAtom, A::MoYeArray{T, N}) where {T, N}
    @assert N ≥ 3
    @assert size(layout(A), 1) == size(get_mma_traits(m).Alayout, 2)
    return make_fragment_like(frgtype_a(get_mma_traits(m)), A) # (V, M, K)
end

function make_fragment_B(m::AbstractMMAAtom, B::MoYeArray{T, N}) where {T, N}
    @assert N ≥ 3
    @assert size(layout(B), 1) == size(get_mma_traits(m).Blayout, 2)
    return make_fragment_like(frgtype_b(get_mma_traits(m)), B) # (V, N, K)
end

struct MMAAtom{Traits} <: AbstractMMAAtom
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

@inline get_mma_traits(mma_atom::MMAAtom) = mma_atom.traits

function Base.show(io::IO, m::MMAAtom)
    println(io, "MMAAtom")
    println(io, "  Thread ID: ", get_mma_traits(m).threadid)
    println(io, "  Layout_A_TV: ", get_mma_traits(m).Alayout)
    println(io, "  Layout_B_TV: ", get_mma_traits(m).Blayout)
    return println(io, "  Layout_C_TV: ", get_mma_traits(m).Clayout)
end

struct TiledMMA{Atom, TiledThr, TiledVal, TiledPerm, TM, TL, TIL} <: AbstractMMAAtom
    mma_atom::Atom
    atom_layout_MNK::TiledThr
    val_layout_MNK::TiledVal
    permutations_MNK::TiledPerm
    tiled_MNK::TM
    thr_layout_VMNK::TL
    tid_layout::TIL
end

@inline get_mma_traits(m::TiledMMA) = get_mma_traits(m.mma_atom)

function TiledMMA(atom::MMAAtom,
                  atom_layout_MNK::Layout{3}=@Layout((1, 1, 1)),
                  val_layout_MNK::Layout{3}=@Layout((1, 1, 1)),
                  permutations_MNK::Tile{3}=(:, :, :))
    MNK = atom.traits.MNK
    threadid = get_mma_traits(atom).threadid

    tiled_MNK = (
        MNK[1] * size(atom_layout_MNK, 1) * size(val_layout_MNK, 1),
        MNK[2] * size(atom_layout_MNK, 2) * size(val_layout_MNK, 2),
        MNK[3] * size(atom_layout_MNK, 3) * size(val_layout_MNK, 3)
    )
    thr_layout_VMNK = tiled_product(threadid, atom_layout_MNK)
    tid_layout = right_inverse(thr_layout_VMNK)
    return TiledMMA{typeof(atom), typeof(atom_layout_MNK), typeof(val_layout_MNK), typeof(permutations_MNK),
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

function thrfrg_C(m::TiledMMA, C::Layout{2})
    tiled_MNK = m.tiled_MNK
    permutations_MNK = m.permutations_MNK
    thr_layout_VMNK = m.thr_layout_VMNK
    atom_MNK = m.mma_atom.traits.MNK

    @assert size(C, 1) % tiled_MNK[1] == Zero()
    @assert size(C, 2) % tiled_MNK[2] == Zero()
    t_tile = (left_inverse(permutations_MNK[1]), left_inverse(permutations_MNK[2]))
    t_array = logical_divide(C, t_tile)

    a_tile = (make_layout(atom_MNK[1]), make_layout(atom_MNK[2]))
    a_array = zipped_divide(t_array, a_tile)

    tv_array = compose(a_array, m.mma_atom.traits.Clayout, :)

    thr_tile = (:, (make_layout(size(thr_layout_VMNK, 2)), make_layout(size(thr_layout_VMNK, 3))))
    thr_array = zipped_divide(tv_array, thr_tile)
    return thr_array
end

function tidfrg_C(m::TiledMMA, C)
    return compose(thrfrg_C(m, C), m.tid_layout, :)
end

function thrfrg_A(m::TiledMMA, A::Layout{2})
    tiled_MNK = m.tiled_MNK
    permutations_MNK = m.permutations_MNK
    thr_layout_VMNK = m.thr_layout_VMNK
    atom_MNK = m.mma_atom.traits.MNK

    @assert size(A, 1) % tiled_MNK[1] == Zero()
    @assert size(A, 2) % tiled_MNK[3] == Zero()

    t_tile = (left_inverse(permutations_MNK[1]), left_inverse(permutations_MNK[3]))
    t_array = logical_divide(A, t_tile)

    a_tile = (make_layout(atom_MNK[1]), make_layout(atom_MNK[3]))
    a_array = zipped_divide(t_array, a_tile)

    tv_array = compose(a_array, m.mma_atom.traits.Alayout, :)
    thr_tile = (:, (make_layout(size(thr_layout_VMNK, 2)), make_layout(size(thr_layout_VMNK, 4))))
    thr_array = zipped_divide(tv_array, thr_tile)
    return thr_array
end

function tidfrg_A(m::TiledMMA, A)
    thr_layout_VMNK = m.thr_layout_VMNK
    atile = (:, (make_layout((size(thr_layout_VMNK, 2), size(thr_layout_VMNK, 3)), (One(), Zero())), :))
    return compose(compose(thrfrg_A(m, A), atile, :), m.tid_layout, :)
end

function thrfrg_B(m::TiledMMA, B::Layout{2})
    tiled_MNK = m.tiled_MNK
    permutations_MNK = m.permutations_MNK
    thr_layout_VMNK = m.thr_layout_VMNK
    atom_MNK = m.mma_atom.traits.MNK

    @assert size(B, 1) % tiled_MNK[2] == Zero()
    @assert size(B, 2) % tiled_MNK[3] == Zero()

    t_tile = (left_inverse(permutations_MNK[2]), left_inverse(permutations_MNK[3]))
    t_array = logical_divide(B, t_tile)

    a_tile = (make_layout(atom_MNK[2]), make_layout(atom_MNK[3]))
    a_array = zipped_divide(t_array, a_tile)

    tv_array = compose(a_array, m.mma_atom.traits.Blayout, :)
    thr_tile = (:, (make_layout(size(thr_layout_VMNK,3)), make_layout(size(thr_layout_VMNK,4))))
    thr_array = zipped_divide(tv_array, thr_tile)
    return thr_array
end

function tidfrg_B(m::TiledMMA, B)
    thr_layout_VMNK = m.thr_layout_VMNK
    btile = (:, (make_layout((size(thr_layout_VMNK, 2), size(thr_layout_VMNK,3)), (One(), Zero())), :))
    return compose(compose(thrfrg_B(m, B), btile, :), m.tid_layout, :)
end

function Base.show(io::IO, m::TiledMMA{Atom, TiledThr, TiledVal, TiledPerm}) where {Atom, TiledThr, TiledVal, TiledPerm}
    println(io, "TiledMMA")
    println(io, "  Tiled Thr: ", TiledThr())
    println(io, "  Tiled Val: ", TiledVal())
    println(io, "  Tiled Perm: ", make_tuple(TiledPerm))
    println(io, "  Tiled MNK: ", m.tiled_MNK)
    println(io, "  Thr Layout VMNK: ", m.thr_layout_VMNK)
    return show(io, m.mma_atom)
end

struct ThrMMA{TA<:TiledMMA, ThrVMNK} <: AbstractMMAAtom
    tiled_mma::TA
    thr_vmnk::ThrVMNK
end

@inline get_mma_traits(m::ThrMMA) = get_mma_traits(m.tiled_mma)

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
    return make_fragment_C(m.tiled_mma.mma_atom, partition_C(m, C))
end

function partiton_fragment_A(m::ThrMMA, A::MoYeArray)
    return make_fragment_A(m.tiled_mma.mma_atom, partition_A(m, A))
end

function partiton_fragment_B(m::ThrMMA, B::MoYeArray)
    return make_fragment_B(m.tiled_mma.mma_atom, partition_B(m, B))
end

function Base.show(io::IO, m::ThrMMA{TA, ThrVMNK}) where {TA, ThrVMNK}
    println(io, "ThrMMA")
    println(io, "  Thr VMNK: ", m.thr_vmnk)
    return show(io, m.tiled_mma)
end

function apply(mma_atom::AbstractMMAAtom, D::MoYeArray{TD, 1}, A::MoYeArray{TA, 1},
               B::MoYeArray{TB, 1}, C::MoYeArray{TC, 1}) where {TD, TA, TB, TC}
    @inline
    return mma_unpack!(get_mma_traits(mma_atom), D, A, B, C)
end
function apply(mma_atom::AbstractMMAAtom, A::MoYeArray, B::MoYeArray, C::MoYeArray)
    @inline
    return apply(mma_atom, C, A, B, C)
end

@inline tile_size(m::TiledMMA, i::IntType) = m.tiled_MNK[i]
@inline tile_size(m::ThrMMA, i::IntType) = tile_size(m.tiled_mma, i)
