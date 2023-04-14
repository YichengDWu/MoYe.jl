module CuTe

include("algorithm/tuple_algorithms.jl")
include("int_tuple.jl")
include("stride.jl")
include("layout.jl")
include("print.jl")

export flatten
export colex_less, elem_less, increment
export coord_to_index, index_to_coord
export Layout, make_layout, shape, rank, depth, cosize, slice, dice, complement,
       logical_product, blocked_product
export print_layout

end
