module MoYe

using Static: StaticInt, static, dynamic, is_static, One, Zero, known, repr
import Static
import ManualMemory, LayoutPointers
import StrideArraysCore
import StaticArrayInterface
using StrideArraysCore: static_length, static_size, static_axes
using StrideArraysCore: @gc_preserve
using CUDA, BFloat16s, LLVM
using CUDA: @device_override
using LLVMLoopInfo: @loopinfo
using Core: LLVMPtr
import Adapt
using MacroTools: @capture
import LinearAlgebra
#import LinearAlgebra

include("utilities.jl")
include("algorithm/tuple_algorithms.jl")
include("int_tuple.jl")
include("stride.jl")
include("layout.jl")
include("engine.jl")

include("array.jl")
include("broadcast.jl")
include("algorithm/array_algorithms.jl")

include("pointer.jl")

# Arch
include("arch/mma/mma.jl")
include("arch/mma/make_mma_ops.jl")
include("arch/copy/copy.jl")
include("arch/copy/copy_async.jl")
include("arch/copy/ldmatrix.jl")

# Traits
include("traits/mma.jl")
include("traits/copy.jl")

include("traits/cp_async.jl")
include("traits/ldmatrix.jl")

# Atom
include("atom/mma.jl")
include("atom/copy.jl")
include("atom/ldmatrix.jl")

include("algorithm/copy.jl")
include("algorithm/blas.jl")

# Device
include("device/smem.jl")

include("print.jl")

# rexport
export static, @gc_preserve, static_size, @loopinfo

# tuple algorithms
export flatten, unflatten
export colex_less, elem_less, increment, capacity
export coord_to_index, index_to_coord, coord_to_coord, compact_col_major, compact_row_major,
       GenColMajor, GenRowMajor, @Layout

# Layout
export Layout, StaticLayout, make_layout, shape, rank, depth, cosize, composition, complement,
       logical_product, blocked_product, raked_product, zipped_product, tiled_product,
       logical_divide, zipped_divide, tiled_divide, zeros!, recast, right_inverse,
       left_inverse, tile_to_shape
export print_layout, print_typst

# MoYeArray
export ArrayEngine, ViewEngine, MoYeArray, make_fragment_like, @parallelize, @tile, zeros!
export MoYeSharedArray

# Traits
export MMATraits, shape_mnk, thr_id, layout_a, layout_b, layout_c, layout_d

# Atom
export CopyAtom, make_tiled_copy, get_slice, partition_D, partition_S, UniversalFMA,
       UniversalCopy, CPOP_ASYNC_CACHEALWAYS, CPOP_ASYNC_CACHEGLOBAL
export MMAAtom, make_tiled_mma, partition_C, partition_A, partition_B, tile_size,
       partition_fragment_C, partition_fragment_A, partition_fragment_B, make_tiled_copy_A,
       make_tiled_copy_B, make_tiled_copy_C, make_fragment_A, make_fragment_B, make_fragment_C,
       retile_D, retile_S

# pointer
export isgmem, issmem, isrmem

# blas
export axpby!, gemm!

# data movement
export cp_async_wait, cp_async_commit

# constants
export _0, _1, _2, _3, _4, _5, _6, _8, _9, _10,
       _16, _32, _64, _128, _256, _512, _1024, _2048, _4096, _8192

end
