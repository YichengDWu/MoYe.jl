abstract type AbstractTraits{OP} end
abstract type AbstractMMATraits{OP <: AbstractMMAOP} <: AbstractTraits{OP} end

# Traits of AbstractMMATraits
# logical types, delegate to mma ops 
@inline valtype_a(::AbstractMMATraits{OP}) where {OP} = valtype_a(OP())
@inline valtype_b(::AbstractMMATraits{OP}) where {OP} = valtype_b(OP())
@inline valtype_c(::AbstractMMATraits{OP}) where {OP} = valtype_c(OP())
@inline valtype_d(::AbstractMMATraits{OP}) where {OP} = valtype_d(OP())

# register types
@inline regtype_a(::AbstractMMATraits{OP}) where {OP} = regtype_a(OP())
@inline regtype_b(::AbstractMMATraits{OP}) where {OP} = regtype_b(OP())
@inline regtype_c(::AbstractMMATraits{OP}) where {OP} = regtype_c(OP())
@inline regtype_d(::AbstractMMATraits{OP}) where {OP} = regtype_d(OP())

# register numbers
@inline regnum_a(::AbstractMMATraits{OP}) where {OP} = regnum_a(OP())
@inline regnum_b(::AbstractMMATraits{OP}) where {OP} = regnum_b(OP())
@inline regnum_c(::AbstractMMATraits{OP}) where {OP} = regnum_c(OP())
@inline regnum_d(::AbstractMMATraits{OP}) where {OP} = regnum_d(OP())

@inline shape_mnk(::AbstractMMATraits{OP}) where {OP} = shape_mnk(OP())

# the following functions need to be implemented for each mma trait
thr_id(::AbstractMMATraits) = error("thr_id not implemented")

# Thr-Val layouts for A, B, C
layout_a(::AbstractMMATraits) = error("layout_a not implemented")
layout_b(::AbstractMMATraits) = error("layout_b not implemented")
layout_c(::AbstractMMATraits) = error("layout_c not implemented")

struct MMATraits{OP <: AbstractMMAOP, D, A, B, C} <: AbstractMMATraits{OP} end
function MMATraits{OP}() where {OP<: AbstractMMAOP}
    return MMATraits{OP, valtype_d(OP()), valtype_a(OP()), valtype_b(OP()), valtype_c(OP())}()
end

@inline shape_mnk(::MMATraits{<:UniversalFMA}) = static((1, 1, 1))
@inline thr_id(::MMATraits{<:UniversalFMA}) = @Layout 1
@inline layout_a(::MMATraits{<:UniversalFMA}) = @Layout (1, 1)
@inline layout_b(::MMATraits{<:UniversalFMA}) = @Layout (1, 1)
@inline layout_c(::MMATraits{<:UniversalFMA}) = @Layout (1, 1)


# utilities
@inline get_regtypes(traits::AbstractMMATraits) = (regtype_d(traits), regtype_a(traits), regtype_b(traits), regtype_c(traits))
@inline get_regnums(traits::AbstractMMATraits) = (regnum_d(traits), regnum_a(traits), regnum_b(traits), regnum_c(traits))

function mma_unpack!(traits::MMATraits{OP, TD, TA, TB, TC},
                     D::LocalArray{TD},
                     A::LocalArray{TA},
                     B::LocalArray{TB},
                     C::LocalArray{TC}) where {OP, TD, TA, TB, TC}
    reg_type_D, reg_type_A, reg_type_B, reg_type_C = get_regtypes(traits)
    reg_num_D, reg_num_A, reg_num_B, reg_num_C = get_regnums(traits)


    rD = recast(reg_type_D, D)
    rA = recast(reg_type_A, A)
    rB = recast(reg_type_B, B)
    rC = recast(reg_type_C, C)

    @assert length(rD) == reg_num_D
    @assert length(rA) == reg_num_A
    @assert length(rB) == reg_num_B
    @assert length(rC) == reg_num_C
    return fma!(mma_op, rD, rA, rB, rC)
end

include("mma_sm70.jl")
include("mma_sm75.jl")
include("mma_sm80.jl")