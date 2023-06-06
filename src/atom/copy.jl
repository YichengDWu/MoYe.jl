abstract type AbstractCopyAtom{Traits, T, OP} <: AbstractCopyTraits{OP} end

struct CopyAtom{Traits, T, OP, VS, VD, VR, NS, ND} <: AbstractCopyAtom{Traits, T, OP}
    traits::Traits
    val_layout_src::VS
    val_layout_dst::VD
    val_layout_ref::VR
    num_val_src::NS
    num_val_dst::ND
end

function Base.getproperty(atom::CopyAtom, x::Symbol)
    if x === :bit_layout_src
        return atom.traits.srclayout
    elseif x === :bit_layout_dst
        return atom.traits.dstlayout
    elseif x === :bit_layout_ref
        return atom.traits.reflayout
    elseif x === :threadid
        return atom.traits.threadid
    else
        return getfield(atom, x)
    end
end

function Base.propertynames(::CopyAtom)
    return (:traits, :bit_layout_src, :bit_layout_dst, :bit_layout_ref, :val_layout_src,
            :val_layout_dst, :val_layout_ref, :num_val_src, :num_val_dst)
end

function CopyAtom{Traits, T}() where {OP, Traits <: CopyTraits{OP}, T}
    @inline
    traits = Traits()
    threadid = traits.threadid

    val_layout_src = upcast(traits.srclayout, sizeof_bits(T))
    val_layout_dst = upcast(traits.dstlayout, sizeof_bits(T))
    val_layout_ref = upcast(traits.reflayout, sizeof_bits(T))

    @assert size(val_layout_src, 1) == size(threadid)
    @assert size(val_layout_dst, 1) == size(threadid)
    @assert size(val_layout_ref, 1) == size(threadid)

    num_val_src = size(val_layout_src, 2)
    num_val_dst = size(val_layout_dst, 2)
    return CopyAtom{typeof(traits), T, OP, typeof(val_layout_src),
                    typeof(val_layout_dst), typeof(val_layout_ref), typeof(num_val_src),
                    typeof(num_val_dst)}(traits, val_layout_src, val_layout_dst,
                                         val_layout_ref, num_val_src, num_val_dst)
end

function CopyAtom{CPOP, T}() where {CPOP <: AbstractCopyOperation, T}
    @inline
    return CopyAtom{CopyTraits{CPOP}, T}()
end

@inline atom(x::CopyAtom) = x

function apply(copy_atom::AbstractCopyAtom, dst::StaticMoYeArray{TD, 1},
               src::StaticMoYeArray{TS, 1}) where {TD, TS}
    if atom(copy_atom).num_val_src == size(src.layout) ||
       atom(copy_atom).num_val_dst == size(dst.layout)
        return copyto_unpack!(copy_atom, dst, src)
    elseif isa(shape(src), Tuple) && isa(shape(dst), Tuple)
        return copyto!(copy_atom, MoYeArray(pointer(dst), layout(dst)[One()]),
                       MoYeArray(pointer(src), layout(src)[One()]))
    else
        throw(ArgumentError("Cannot copy from $src to $dst"))
    end
end

function apply(copy_atom::AbstractCopyAtom, dst::MoYeArray{TD, 1, ED, <:Layout{1, <:Tuple}},
               src::MoYeArray{TS, 1, ES, <:Layout{1, <:Tuple}}) where {TD, TS, ED, ES}
    return copyto!(copy_atom, MoYeArray(pointer(dst), layout(dst)[One()]),
                   MoYeArray(pointer(src), layout(src)[One()]))
end

struct TiledCopy{Traits, T, OP, CP, LT, ST} <: AbstractCopyAtom{Traits, T, OP}
    copy_atom::CP
    tiled_layout_TV::LT
    tiler_MN::ST
end

function Base.getproperty(tiled_copy::TiledCopy, x::Symbol)
    if x === :atom_threadid
        return tiled_copy.copy_atom.threadid
    elseif x === :atom_layout_src
        return tiled_copy.copy_atom.val_layout_src
    elseif x === :atom_layout_dst
        return tiled_copy.copy_atom.val_layout_dst
    elseif x === :atom_layout_ref
        return tiled_copy.copy_atom.val_layout_ref
    else
        return getfield(tiled_copy, x)
    end
end

function Base.propertynames(::TiledCopy)
    return (:copy_atom, :tiled_layout_TV, :tiler_MN, :atom_threadid, :atom_layout_src,
            :atom_layout_dst, :atom_layout_ref)
end

@inline atom(tile_copy::TiledCopy) = atom(tile_copy.copy_atom)

function TiledCopy(atom::CopyAtom{Traits, T, OP}, tiled_layout_TV::Layout,
                   tiler_MN) where {Traits, T, OP}
    @assert size(tiled_layout_TV, 1) % size(atom.val_layout_ref, 1) == Zero()
    @assert size(tiled_layout_TV, 2) % size(atom.val_layout_ref, 2) == Zero()
    return TiledCopy{Traits, T, OP, typeof(atom), typeof(tiled_layout_TV), typeof(tiler_MN)
                     }(atom, tiled_layout_TV, tiler_MN)
end

function tile2thrfrg(tiled_copy::TiledCopy, x::Layout, ref2trg)
    atom_num_thr = size(tiled_copy.copy_atom.val_layout_ref, 1)
    atom_num_val = size(tiled_copy.copy_atom.val_layout_ref, 2)
    atom_layout_TV = zipped_divide(tiled_copy.tiled_layout_TV, (atom_num_thr, atom_num_val))
    trg_layout_TV = compose(atom_layout_TV, ref2trg, :)
    thrval2mn = coalesce(_zip(trg_layout_TV), (One(), (One(), One())))
    tv_array = compose(x, thrval2mn, :)
    return tv_array((:, :), :)
end

function tidfrg_S(tiled_copy::TiledCopy, src::Layout{N}) where {N}
    @assert N>=rank(shape(tiled_copy.tiler_MN)) "The dimension is too small to be tiled."
    return tile2thrfrg(tiled_copy, zipped_divide(src, tiled_copy.tiler_MN),
                       compose(right_inverse(tiled_copy.copy_atom.val_layout_ref),
                               tiled_copy.copy_atom.val_layout_src))
end

function tidfrg_D(tiled_copy::TiledCopy, dst::Layout{N}) where {N}
    @assert N>=rank(shape(tiled_copy.tiler_MN)) "The dimension is too small to be tiled."
    return tile2thrfrg(tiled_copy, zipped_divide(dst, tiled_copy.tiler_MN),
                       compose(right_inverse(tiled_copy.copy_atom.val_layout_ref),
                               tiled_copy.copy_atom.val_layout_dst))
end

function retile(tiled_copy, x::StaticMoYeArray{T, R}) where {T, R}
    V = size(layout(x), 1)
    tiled_layout_TV = tiled_copy.tiled_layout_TV
    tiled_shape_MN = shape(tiled_copy.tiler_MN)
    atom_num_val = size(tiled_copy.copy_atom.val_layout_ref, 2)
    tiled_num_thr = size(tiled_layout_TV, 1)
    frg_layout_mn = upcast(composition(right_inverse(tiled_layout_TV),
                                       make_layout(tiled_shape_MN)), tiled_num_thr * V)
    frg_layout_v = zipped_divide(logical_product(make_layout(V),
                                                 right_inverse(frg_layout_mn)),
                                 make_layout(atom_num_val))

    t_array = zipped_divide(x, prepend(product_each(shape(frg_layout_mn)), V))
    v_array = compose(t_array, frg_layout_v, :)
    return view(v_array, :, append(One(), :, StaticInt{R}()))
end

function get_layoutS_TV(tiled_copy::TiledCopy)
    ref_S = make_layout((shape(tiled_copy.tiler_MN), One()))
    return tile2thrfrg(tiled_copy, ref_S,
                       composition(right_inverse(tiled_copy.copy_atom.val_layout_ref),
                                   tiled_copy.copy_atom.val_layout_src))(:, :, One())
end

function get_layoutS_MN(tiled_copy::TiledCopy)
    tiled_shape_MN = shape(tiled_copy.tiler_MN)
    layoutS_TV = get_layoutS_TV(tiled_copy)
    layoutS_MK = composition(right_inverse(layoutS_TV), make_layout(tiled_shape_MN))
    thrID_S = make_layout(size(tiled_copy.tiled_layout_TV, 1))
    return (layoutS_MK, thrID_S)
end

function get_layoutD_TV(tiled_copy::TiledCopy)
    ref_D = make_layout((shape(tiled_copy.tiler_MN), One()))
    return tile2thrfrg(tiled_copy, ref_D,
                       compose(right_inverse(tiled_copy.copy_atom.val_layout_ref),
                               tiled_copy.copy_atom.val_layout_dst))(:, :, One())
end

function get_layoutD_MN(tiled_copy::TiledCopy)
    tiled_shape_MN = shape(tiled_copy.tiler_MN)
    layoutD_TV = get_layoutD_TV(tiled_copy)
    layoutD_MK = composition(right_inverse(layoutD_TV), make_layout(tiled_shape_MN))
    thrID_D = make_layout(size(tiled_copy.tiled_layout_TV, 1))
    return (layoutD_MK, thrID_D)
end

struct ThrCopy{Traits, T, OP, TC, TI} <: AbstractCopyAtom{Traits, T, OP}
    tiled_copy::TC
    thr_idx::TI
    function ThrCopy(tiled_copy::TiledCopy{Traits, T, OP}, thr_idx) where {Traits, T, OP}
        return new{Traits, T, OP, typeof(tiled_copy), typeof(thr_idx)}(tiled_copy, thr_idx)
    end
end

@inline atom(thr_copy::ThrCopy) = atom(thr_copy.tiled_copy)

function partition_S(thr_copy::ThrCopy, s::MoYeArray{T, N}) where {T, N}
    thr_tensor = MoYeArray(pointer(s), tidfrg_S(thr_copy.tiled_copy, layout(s)))
    return view(thr_tensor, thr_copy.thr_idx, :, repeat(:, StaticInt{N}()))
end

function partition_D(thr_copy::ThrCopy, d::MoYeArray{T, N}) where {T, N}
    thr_tensor = MoYeArray(pointer(d), tidfrg_D(thr_copy.tiled_copy, layout(d)))
    return view(thr_tensor, thr_copy.thr_idx, :, repeat(:, StaticInt{N}()))
end

function retile_S(thr_copy::ThrCopy, s::StaticMoYeArray{T, R}) where {T, R}
    return MoYeArray(pointer(s), retile(thr_copy.tiled_copy, layout(s)))
end

function retile_D(thr_copy::ThrCopy, d::StaticMoYeArray{T, R}) where {T, R}
    return MoYeArray(pointer(d), retile(thr_copy.tiled_copy, layout(d)))
end

@inline get_slice(tiled_copy::TiledCopy, thr_idx) = ThrCopy(tiled_copy, thr_idx)
@inline get_thread_slice(tiled_copy::TiledCopy, thr_idx) = get_slice(tiled_copy, thr_idx)

function make_tiled_copy(copy_atom::CopyAtom, thr_layout::Layout{TR},
                         val_layout::Layout{TV}=@Layout(1)) where {TR, TV}
    R = max(TR, TV)

    thr_layout_mn = append(thr_layout, @Layout(1), StaticInt{R}())
    val_layout_mn = append(val_layout, @Layout(1), StaticInt{R}())

    layout_mn = raked_product(thr_layout_mn, val_layout_mn, true)
    layout_tv = composition(right_inverse(layout_mn),
                            make_layout((size(thr_layout), size(val_layout))))
    return TiledCopy(copy_atom, layout_tv, product_each(shape(layout_mn)))
end

function tile_size(tiled_copy::TiledCopy)
    return shape(tiled_copy.tiler_MN)
end

function Base.size(tiled_copy::TiledCopy)
    return size(tiled_copy.tiled_layout_TV, 1)
end

function Base.show(io::IO, m::CopyAtom{Traits, T}) where {Traits, T}
    println(io, "CopyAtom")
    println(io, "  Thread ID: ", m.traits.threadid)
    println(io, "  ValLayoutSrc: ", m.val_layout_src)
    println(io, "  ValLayoutDst: ", m.val_layout_dst)
    println(io, "  ValLayoutRef: ", m.val_layout_ref)
    return println(io, "  ValueType:    $(Int(sizeof_bits(T)))b")
end

function Base.show(io::IO, m::TiledCopy)
    println(io, "TiledCopy")
    println(io, "  Tiler_MN:    ", m.tiler_MN)
    println(io, "  TiledLayout_TV: ", m.tiled_layout_TV)
    return show(io, m.copy_atom)
end

function Base.show(io::IO, m::ThrCopy)
    println(io, "ThrCopy")
    println(io, "  ThrIdx: ", m.thr_idx)
    return show(io, m.tiled_copy)
end
