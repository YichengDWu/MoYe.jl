module MoYe

using Static: StaticInt, IntType, static, dynamic, is_static, One, Zero
import Static
import ManualMemory, LayoutPointers
import StrideArraysCore
import StaticArrayInterface
using StrideArraysCore: static_length, static_size, static_axes
using StrideArraysCore: @gc_preserve
using CUDA, BFloat16s, LLVM
using CUDA: @device_override
using LLVMLoopInfo
using Core: LLVMPtr
import Adapt

include("utilities.jl")
include("algorithm/tuple_algorithms.jl")
include("int_tuple.jl")
include("stride.jl")
include("layout.jl")
include("print.jl")
include("engine.jl")

include("array.jl")
include("broadcast.jl")
include("algorithm/array_algorithms.jl")

include("pointer.jl")
include("device/smem.jl")

# Arch
include("arch/mma/mma.jl")
include("arch/mma/mma_sm80.jl")
include("arch/copy/copy.jl")
include("arch/copy/copy_async.jl")
include("arch/copy/ldmatrix.jl")

# Traits
include("traits/mma.jl")
include("traits/mma_sm80.jl")
include("traits/copy.jl")
include("traits/cp_async.jl")
include("traits/ldmatrix.jl")

# Atom
include("atom/mma.jl")
include("atom/copy.jl")

include("algorithm/copy.jl")
include("algorithm/blas.jl")

# Deprecations
include("deprecated.jl")

# rexport
export static, @gc_preserve

# tuple algorithms
export flatten
export colex_less, elem_less, increment, capacity
export coord_to_index, index_to_coord, coord_to_coord, compact_col_major, compact_row_major,
       GenColMajor, GenRowMajor, @Layout, make_tuple

# Layout
export Layout, make_layout, shape, rank, depth, cosize, compose, complement, logical_product,
       blocked_product, raked_product, zipped_product, logical_divide, zipped_divide,
       tiled_divide, zeros!, recast, right_inverse, left_inverse
export print_layout

# MoYeArray
export ArrayEngine, ViewEngine, MoYeArray, make_fragment_like, @parallelize, @tile, zeros!
export MoYeSharedArray

# Atom
export CopyAtom, make_tiled_copy, get_thread_slice, partition_D, partition_S
export MMAAtom, make_tiled_mma, partition_C, partition_A, partition_B

# pointer
export isgmem, issmem, isrmem

# blas
export axpby!

# data movement
export cucopyto!, cp_async_wait, cp_async_commit

end
