module CuTe

include("algorithm/tuple_algorithms.jl")
include("int_tuple.jl")
include("stride.jl")
include("layout.jl")

export flatten
export colex_less, elem_less, increment
export coord_to_index, index_to_coord
export Layout, make_layout, shape, stride, rank, depth, cosize, slice, dice

end
