module CuTe

using Static: StaticInt, IntType, static

include("algorithm/tuple_algorithms.jl")
include("int_tuple.jl")
include("stride.jl")
include("layout.jl")
include("print.jl")

export flatten
export colex_less, elem_less, increment, capacity
export coord_to_index, index_to_coord
export Layout, make_layout, shape, rank, depth, cosize, complement, logical_product,
       blocked_product, raked_product, logical_divide, zipped_divide
export print_layout

end
