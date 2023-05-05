#=
struct MMAAtom{Traits <: MMATraits, DFrgType, AFrgType, BFrgType, CFrgType}
    traits::Traits
end

function MMAAtom{Traits}() where {Traits <: MMATraits}
    DFrgType = fragtype_d(Traits)
    AFrgType = fragtype_a(Traits)
    BFrgType = fragtype_b(Traits)
    CFrgType = fragtype_c(Traits)
    return MMAAtom{Traits, DFrgType, AFrgType, BFrgType, CFrgType}(Traits)
end

function MMAAtom{OP}() where {OP <: MMAOP}
    return MMAAtom{MMATraits{OP}}()
end

function mma_unpack(D::MoYeArray{DT, 1}, A::MoYeArray{AT, 1}, B::MoYeArray{BT, 1},
                    C::MoYeArray{CT, 1}, ::MMAAtom) end
function mma_unpack(A::MoYeArray{AT, 1}, B::MoYeArray{BT, 1}, C::MoYeArray{CT, 1},
                    ::MMAAtom)
    return mma_unpack(C, A, B, C)
end

@inline function make_fragment_A(A::MoYeArray,
                                 ::MMAAtom{Traits, DFrgType, AFrgType}) where {Traits,
                                                                               DFrgType,
                                                                               AFrgType}
    #@assert rank(A) ≥ static(3)
    #@assert size(A.layout, 1) == size(m.traits.Alayout, 2)
    return make_fragment_like(AFrgType, A)     # When would AFrgType be a pointer?
end

@inline function make_fragment_B(B::MoYeArray,
                                 ::MMAAtom{Traits, DFrgType, AFrgType, BFrgType}) where {
                                                                                         Traits,
                                                                                         DFrgType,
                                                                                         AFrgType,
                                                                                         BFrgType
                                                                                         }
    #@assert rank(B) ≥ static(3)
    #@assert size(B.layout, 1) == size(m.traits.Blayout, 2)
    return make_fragment_like(BFrgType, B)
end

@inline function make_fragment_C(C::MoYeArray,
                                 m::MMAAtom{Traits, DFrgType, AFrgType, BFrgType}) where {
                                                                                          Traits,
                                                                                          DFrgType,
                                                                                          AFrgType,
                                                                                          BFrgType,
                                                                                          CFrgType
                                                                                          }
    #@assert rank(C) ≥ static(3)
    #@assert size(C.layout, 1) == size(m.traits.Clayout, 2)
    return make_fragment_like(CFrgType, shape(C))
end
=#
