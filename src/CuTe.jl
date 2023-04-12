module CuTe

include("algorithm/tuple_algorithms.jl")
include("int_tuple.jl")
include("stride.jl")

export flatten
export colex_less, elem_less, increment
export coord_to_index
end
