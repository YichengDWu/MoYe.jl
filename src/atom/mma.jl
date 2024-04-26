abstract type AbstractMMAAtom{Traits} end

# Atom interface
@inline valtype_a(::AbstractMMAAtom{Traits}) where {Traits} = valtype_a(Traits())
@inline valtype_b(::AbstractMMAAtom{Traits}) where {Traits} = valtype_b(Traits())
@inline valtype_c(::AbstractMMAAtom{Traits}) where {Traits} = valtype_c(Traits())
@inline valtype_d(::AbstractMMAAtom{Traits}) where {Traits} = valtype_d(Traits())

@inline regtype_a(::AbstractMMAAtom{Traits}) where {Traits} = regtype_a(Traits())
@inline regtype_b(::AbstractMMAAtom{Traits}) where {Traits} = regtype_b(Traits())
@inline regtype_c(::AbstractMMAAtom{Traits}) where {Traits} = regtype_c(Traits())
@inline regtype_d(::AbstractMMAAtom{Traits}) where {Traits} = regtype_d(Traits())

#@inline regnum_a(::AbstractMMAAtom{OP}) where {OP} = regnum_a(OP())
#@inline regnum_b(::AbstractMMAAtom{OP}) where {OP} = regnum_b(OP())
#@inline regnum_c(::AbstractMMAAtom{OP}) where {OP} = regnum_c(OP())
#@inline regnum_d(::AbstractMMAAtom{OP}) where {OP} = regnum_d(OP())

@inline layout_a(::AbstractMMAAtom{Traits}) where {Traits} = layout_a(Traits())
@inline layout_b(::AbstractMMAAtom{Traits}) where {Traits} = layout_b(Traits())
@inline layout_c(::AbstractMMAAtom{Traits}) where {Traits} = layout_c(Traits())

@inline thr_id(::AbstractMMAAtom{Traits}) where {Traits} = thr_id(Traits())
@inline shape_mnk(::AbstractMMAAtom{Traits}) where {Traits} = shape_mnk(Traits())

function make_fragment_C(m::AbstractMMAAtom, C::MoYeArray{T, N}) where {T, N}
    @inline
    @assert N ≥ 3
    @assert size(layout(C), 1) == size(layout_c(m), 2)
    return MoYeArray{regtype_c(m)}(undef, shape(C)) # (V, M, N)
end

# Note that hopper architecture needs to return a view of A/B for the fragment
# In this case we always have regtype_a(m) == T
function make_fragment_A(m::AbstractMMAAtom, A::MoYeArray{T, N}) where {T, N}
    @inline
    @assert N ≥ 3
    @assert size(layout(A), 1) == size(layout_a(m), 2)
    return make_fragment_like(regtype_a(m), A) # (V, M, K)
end

function make_fragment_B(m::AbstractMMAAtom, B::MoYeArray{T, N}) where {T, N}
    @inline
    @assert N ≥ 3
    @assert size(layout(B), 1) == size(layout_b(m), 2)
    return make_fragment_like(regtype_b(m), B) # (V, N, K)
end

struct MMAAtom{Traits} <: AbstractMMAAtom{Traits} 
    function MMAAtom{Traits}(args...) where {Traits <: AbstractMMATraits}
        traits = Traits(args...)
        return new{typeof(traits)}()
    end
    function MMAAtom{OP}(args...) where {OP <: AbstractMMAOP}
        traits = MMATraits{OP}(args...)
        return new{typeof(traits)}()
    end
end

function Base.show(io::IO, m::MMAAtom)
    println(io, "MMAAtom")
    println(io, "  Thread ID: ", thr_id(m))
    println(io, "  Layout_A_TV: ", layout_a(m))
    println(io, "  Layout_B_TV: ", layout_b(m))
    return println(io, "  Layout_C_TV: ", layout_c(m))
end

function apply(mma_atom::AbstractMMAAtom{Traits}, D::MoYeArray{TD, 1}, A::MoYeArray{TA, 1},
               B::MoYeArray{TB, 1}, C::MoYeArray{TC, 1}) where {Traits, TD, TA, TB, TC}
    @inline
    return mma_unpack!(Traits(), D, A, B, C)
end
function apply(mma_atom::AbstractMMAAtom, A::MoYeArray, B::MoYeArray, C::MoYeArray)
    @inline
    return apply(mma_atom, C, A, B, C)
end

# TiledMMA 
struct TiledMMA{Atom<:AbstractMMAAtom, 
              AtomLayoutMNK <: Layout{3},
              PermutationMNK <: Tile{3}}
    atom::Atom
    atom_layout_mnk::AtomLayoutMNK
    permutation_mnk::PermutationMNK
end

function TiledMMA{Atom, AtomLayoutMNK, PermutationMNK}() where {Atom, AtomLayoutMNK, PermutationMNK}
    return TiledMMA{Atom, AtomLayoutMNK, PermutationMNK}(Atom(), AtomLayoutMNK(), make_tuple(PermutationMNK))
end

function Base.show(io::IO, m::TiledMMA)
    println(io, "TiledMMA")
    println(io, "  ThrLayoutVMNK: ", get_thr_layout_vmnk(m))
    println(io, "  PermutationMNK: ", m.permutation_mnk)
    return show(io, m.atom)
end

struct ThrMMA{TA<:TiledMMA, ThrVMNK}
    tiled_mma::TA
    thr_vmnk::ThrVMNK
end

function Base.show(io::IO, m::ThrMMA{TA, ThrVMNK}) where {TA, ThrVMNK}
    println(io, "ThrMMA")
    println(io, "  Thr VMNK: ", m.thr_vmnk)
    return show(io, m.tiled_mma)
end

# all the delegation functions
@inline layout_a(t::TiledMMA) = layout_a(t.atom)
@inline layout_b(t::TiledMMA) = layout_b(t.atom)
@inline layout_c(t::TiledMMA) = layout_c(t.atom)
@inline thr_id(t::TiledMMA) = thr_id(t.atom)
@inline shape_mnk(t::TiledMMA) = shape_mnk(t.atom)

@inline @generated function get_thr_layout_vmnk(::TiledMMA{Atom, AtomLayoutMNK}) where {Atom, AtomLayoutMNK}
    thr_layout_vmnk_ = tiled_product(thr_id(Atom()), AtomLayoutMNK()) # (thr_id, atom_M, atom_N, atom_K)
    return :($thr_layout_vmnk_)
end

function thrfrg_C(m::TiledMMA, C::Layout{2})
    thr_layout_vmnk = get_thr_layout_vmnk(m)
    atom_mnk = shape_mnk(m.atom)
    permutation_mnk = m.permutation_mnk

    # Assume C > permutation_mn and permutation_mn divides C such logical_divide makes sense
    # Because of the assumption, the "effective" tiledmma size scales up
    # If permutation_mnk is a tuple of ints then there is no actual permutation, it simply add nested sublayouts to the layout
    t_array = logical_divide(C, (permutation_mnk[1], permutation_mnk[2]))
    a_array = zipped_divide(t_array, map(make_layout, (atom_mnk[1], atom_mnk[2])))  # ((atom_m, atom_n), (rest_m, rest_n))
    tv_array = composition(a_array, (layout_c(m), :))                     # ((thr_id, val_idx), (rest_m, rest_n))

    thr_tile = (:, (make_layout(size(thr_layout_vmnk, _2)), make_layout(size(thr_layout_vmnk, _3)))) # (:, (atom_M, atom_N))
    
    # ((thr_id, val_idx), ((atom_M, atom_N), (rest_m/atom_M, rest_n/atom_N)) -> ((thr_id, (atom_M, atom_N)), (val_idx, (rest_m/atom_M, rest_n/atom_N)))
    thr_array = zipped_divide(tv_array, thr_tile) 
    return thr_array
end

function thrfrg_A(m::TiledMMA, A::Layout{2})
    thr_layout_vmnk = get_thr_layout_vmnk(m)
    atom_mnk = shape_mnk(m.atom)
    permutation_mnk = m.permutation_mnk

    t_array = logical_divide(A, (permutation_mnk[1], permutation_mnk[3])) 
    a_array = zipped_divide(t_array, map(make_layout, map(make_layout, (atom_mnk[1], atom_mnk[3])))) 
    tv_array = composition(a_array, (layout_a(m), :))                   

    thr_tile = (:, (make_layout(size(thr_layout_vmnk, _2)), make_layout(size(thr_layout_vmnk, _4)))) # (:, (thrM, thrK))
    
    # ((thr_id, val_idx), ((M, K), (rest_m/M, rest_k/K)) -> ((thr_id, (thrM, thrK)), (val_idx, (rest_m/thrM, rest_k/thrK)))
    thr_array = zipped_divide(tv_array, thr_tile) 
    return thr_array
end

function thrfrg_B(m::TiledMMA, B::Layout{2})
    thr_layout_vmnk = get_thr_layout_vmnk(m)
    atom_mnk = shape_mnk(m.atom)
    permutation_mnk = m.permutation_mnk

    t_array = logical_divide(B, (permutation_mnk[2], permutation_mnk[3])) 
    b_array = zipped_divide(t_array, map(make_layout, map(make_layout, (atom_mnk[2], atom_mnk[3])))) 
    tv_array = composition(b_array, (layout_b(m), :))                   

    thr_tile = (:, (make_layout(size(thr_layout_vmnk, _3)), make_layout(size(thr_layout_vmnk, _4)))) # (:, (thrN, thrK))
    
    # ((thr_id, val_idx), ((N, K), (rest_n/N, rest_k/K)) -> ((thr_id, (thrN, thrK)), (val_idx, (rest_n/thrN, rest_k/thrK)))
    thr_array = zipped_divide(tv_array, thr_tile) 
    return thr_array
end

function get_slice(m::TiledMMA, thr_idx::DInt)
    @inline
    thr_vmnk = get_congr_coord(get_thr_layout_vmnk(m), thr_idx)
    return ThrMMA(m, thr_vmnk)
end

@inline Base.size(x::TiledMMA) = size(get_thr_layout_vmnk(x))
@generated function tile_size(m::TiledMMA, ::StaticInt{I}) where {I}
    @assert I in 1:3
    m = m()
    core_size = shape_mnk(m)[I] * size(get_thr_layout_vmnk(m), I+1)
    s = m.permutation_mnk[I]
    perm_size = s isa Layout ? size(s) : s
    if perm_size isa Colon
        return :($core_size)
    else
        return :($(max(core_size, perm_size)))
    end
end

@inline tile_shape(m::TiledMMA) = (tile_size(m, One()), tile_size(m, _2), tile_size(m, _3))

function make_tiled_mma(mma_atom::AbstractMMAAtom,
                        atom_layout::Layout=@Layout((1, 1, 1)),
                        permutations::Tile=(:, :, :))
    atom_layout_mnk = append(atom_layout, @Layout(1, 0), _3)
    permutation_mnk = append(permutations, :, _3)
    return TiledMMA(mma_atom, atom_layout_mnk, permutation_mnk)
end

"""
    make_tiled_mma(mma_op, atom_layout, permutations)

Create a TiledMMA object from an MMA operation, atom layout, and permutations.
See also [`print_typst`](@ref).

## Arguments

  - `mma_op::OP`: The MMA operation.
  - `atom_layout::Layout`: The layout of the atom.
  - `permutations::Tile`: The permutations of the atom.

## Examples

```julia
julia> tiled_mma = make_tiled_mma(MMAOP_8x8x4_F32F16F16F32_NT(), @Layout((2,2), (2,1)), (@Layout((4,4,2), (1,8,4)), _32, _4))
TiledMMA
  ThrLayoutVMNK: ((_4, _2), _2, _2, _1):((_1, _16), _8, _4, _0)
  PermutationMNK: ((_4, _4, _2):(_1, _8, _4), _32, _4)
MMAAtom
  Thread ID: (_4, _2):(_1, _16)
  Layout_A_TV: ((_4, _2), _4):((_8, _4), _1)
  Layout_B_TV: ((_4, _2), _4):((_8, _4), _1)
  Layout_C_TV: ((_2, _2, _2), (_2, _2, _2)):((_1, _16, _4), (_8, _2, _32))

```

"""
function make_tiled_mma(mma_op::OP,
                        atom_layout::Layout=@Layout((1, 1, 1)),
                        permutations::Tile=(:, :, :)) where {OP<: AbstractMMAOP}
    return make_tiled_mma(MMAAtom{OP}(), atom_layout, permutations)
end

function get_layoutC_MN(tiled_mma::TiledMMA)
    ref_C = make_layout((tile_size(tiled_mma, One()), tile_size(tiled_mma, _2)))
    layoutC_TV = thrfrg_C(tiled_mma, ref_C)
    layoutC_MN = withshape(right_inverse(layoutC_TV), shape(ref_C))

    thrid_C = get_thr_layout_vmnk(tiled_mma)(:, :, :, Zero())
    return layoutC_MN, thrid_C
end

function get_layoutA_MK(tiled_mma::TiledMMA)
    ref_A = make_layout((tile_size(tiled_mma, One()), tile_size(tiled_mma, _3)))
    layoutA_TV = thrfrg_A(tiled_mma, ref_A)
    layoutA_MK = withshape(right_inverse(layoutA_TV), shape(ref_A))

    thrid_A = get_thr_layout_vmnk(tiled_mma)(:, :, Zero(), :)
    return layoutA_MK, thrid_A
end

function get_layoutB_NK(tiled_mma::TiledMMA)
    ref_B = make_layout((tile_size(tiled_mma, _2), tile_size(tiled_mma, _3)))
    layoutB_TV = thrfrg_B(tiled_mma, ref_B)
    layoutB_NK = withshape(right_inverse(layoutB_TV), shape(ref_B))

    thrid_B = get_thr_layout_vmnk(tiled_mma)(:, Zero(), :, :)
    return layoutB_NK, thrid_B
end

@inline tile_size(m::ThrMMA, i::IntType) = tile_size(m.tiled_mma, i)

function get_layoutC_TV(tiled_mma::TiledMMA)
    ref_C = make_layout((tile_size(tiled_mma, One()), tile_size(tiled_mma, _2)))
    layoutC_TV = thrfrg_C(tiled_mma, ref_C)
    thridx_to_thrid = right_inverse(get_thr_layout_vmnk(tiled_mma))
    return composition(layoutC_TV, (thridx_to_thrid, :))
end

function get_layoutA_TV(tiled_mma::TiledMMA)
    ref_A = make_layout((tile_size(tiled_mma, One()), tile_size(tiled_mma, _3)))
    layoutA_TV = thrfrg_A(tiled_mma, ref_A)
    thr_layout_vmnk = get_thr_layout_vmnk(tiled_mma)
    # insert N dimension to reflect the projection in A
    atile = (:, (make_layout((size(thr_layout_vmnk, _2), size(thr_layout_vmnk, _3)), (One(), Zero())), :))
    thridx_to_thrid = right_inverse(thr_layout_vmnk)
    return composition(composition(layoutA_TV, (atile, :)), (thridx_to_thrid, :))
end

function get_layoutB_TV(tiled_mma::TiledMMA)
    ref_B = make_layout((tile_size(tiled_mma, _2), tile_size(tiled_mma, _3)))
    layoutB_TV = thrfrg_B(tiled_mma, ref_B)
    thr_layout_vmnk = get_thr_layout_vmnk(tiled_mma)
    # insert M dimension to reflect the projection in B
    btile = (:, (make_layout((size(thr_layout_vmnk, _2), size(thr_layout_vmnk, _3)), (One(), Zero())), :))
    thridx_to_thrid = right_inverse(thr_layout_vmnk)
    return composition(composition(layoutB_TV, (btile, :)), (thridx_to_thrid, :))
end


function partition_C(m::ThrMMA, C::MoYeArray)
    thr_array = MoYeArray(pointer(C), thrfrg_C(m.tiled_mma, layout(C)))
    thr_vmn = (m.thr_vmnk[1], (m.thr_vmnk[2], m.thr_vmnk[3])) # (V, (M, N))
    return view(thr_array, thr_vmn, (:, repeat(:, rank(layout(thr_array)[2][2]))))
end

function partition_A(m::ThrMMA, A::MoYeArray)   
    thr_array = MoYeArray(pointer(A), thrfrg_A(m.tiled_mma, layout(A)))
    thr_vmk = (m.thr_vmnk[1], (m.thr_vmnk[2], m.thr_vmnk[4])) # (V, (M, K))
    return view(thr_array, thr_vmk, (:, repeat(:, rank(layout(thr_array)[2][2]))))  
end

function partition_B(m::ThrMMA, B::MoYeArray)
    thr_array = MoYeArray(pointer(B), thrfrg_B(m.tiled_mma, layout(B)))
    thr_vnk = (m.thr_vmnk[1], (m.thr_vmnk[3], m.thr_vmnk[4])) # (V, (N, K))
    return view(thr_array, thr_vnk, (:, repeat(:, rank(layout(thr_array)[2][2]))))
end

function partition_fragment_C(m::ThrMMA, C::MoYeArray)
    @inline
    return make_fragment_C(m.tiled_mma.atom, partition_C(m, C))
end

function partition_fragment_A(m::ThrMMA, A::MoYeArray)
    @inline
    return make_fragment_A(m.tiled_mma.atom, partition_A(m, A))
end

function partition_fragment_B(m::ThrMMA, B::MoYeArray)
    @inline
    return make_fragment_B(m.tiled_mma.atom, partition_B(m, B))
end

function partition_shape_C(m::TiledMMA, shape_MN::StaticIntTuple{R}) where {R}
    @assert R >= 2
    atom_mnk = shape_mnk(m)
    V = shape(layout_c(m))[2]
    M = shape_div(shape_MN[1], atom_mnk[1]* m.thr_vmnk[2])
    N = shape_div(shape_MN[2], atom_mnk[2]* m.thr_vmnk[3])
    return (V, M, N, shape_MN[3:R]...)
end

function partition_fragment_C(m::TiledMMA, shape_MN::StaticIntTuple)
    @inline
    return MoYeArray{regtype_c(m)}(undef, partition_shape_C(m, shape_MN))
end