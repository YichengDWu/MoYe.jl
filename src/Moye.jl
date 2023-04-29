module MoYe

using Static: StaticInt, IntType, static, dynamic, is_static, One, Zero
import Static
import ManualMemory, LayoutPointers
import StrideArraysCore
using StrideArraysCore: @gc_preserve
using CUDA, BFloat16s, LLVM
using KernelAbstractions.Extras: @unroll
using Core: LLVMPtr
import Adapt

include("utilities.jl")
include("algorithm/tuple_algorithms.jl")
include("int_tuple.jl")
include("stride.jl")
include("layout.jl")
include("print.jl")
include("container/engine.jl")

include("array.jl")
include("broadcast.jl")
include("algorithm/array_algorithms.jl")
include("algorithm/blas.jl")

include("pointer.jl")
include("device/memory.jl")

include("arch/mma.jl")
include("arch/copy/copy.jl")
include("arch/copy/copy_async.jl")
include("arch/copy/ldmatrix.jl")

include("atom/mma_traits.jl")
include("atom/copy/copy_traits.jl")

include("algorithm/copy.jl")

# rexport
export static, @gc_preserve

# tuple algorithms
export flatten
export colex_less, elem_less, increment, capacity
export coord_to_index, index_to_coord, coord_to_coord, compact_col_major, compact_row_major,
       GenColMajor, GenRowMajor, @Layout, make_tuple

# layout
export Layout, make_layout, shape, rank, depth, cosize, complement, logical_product,
       blocked_product, raked_product, zipped_product, logical_divide, zipped_divide,
       tiled_divide, local_partition, local_tile, zeros!, recast, right_inverse
export print_layout

# MoYeArray
export ArrayEngine, ViewEngine, MoYeArray, make_fragment_like

# pointer
export isgmem, issmem, isrmem

# blas
export axpby!

# data movement
export cucopyto!

end
