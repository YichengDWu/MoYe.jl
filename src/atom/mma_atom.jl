struct MMAAtom{Traits<: MMATraits, DFrgType, AFrgType, BFrgType, CFrgType}
    traits::Traits
end

function MMAAtom{Traits}() where {Traits<: MMATraits}
    DFrgType = fragtype_d(Traits)
    AFrgType = fragtype_a(Traits)
    BFrgType = fragtype_b(Traits)
    CFrgType = fragtype_c(Traits)
    MMAAtom{Traits, DFrgType, AFrgType, BFrgType, CFrgType}(Traits)
end

function MMAAtom{OP}() where {OP<:MMAOP}
    MMAAtom{MMATraits{OP}}()
end

function mma_unpack(D::CuTeArray{DT, 1}, A::CuTeArray{AT, 1} ,
                    B::CuTeArray{BT, 1}, C::CuTeArray{CT, 1}, ::MMAAtom)

end
function mma_unpack(A::CuTeArray{AT, 1} , B::CuTeArray{BT, 1}, C::CuTeArray{CT, 1}, ::MMAAtom)
    mma_unpack(C, A, B, C)
end

@inline function make_fragment_A(A::CuTeArray, ::MMAAtom{Traits, DFrgType, AFrgType}) where {Traits, DFrgType, AFrgType}
    return make_fragment_like(AFrgType, A)
end

@inline function make_fragment_B(B::CuTeArray, ::MMAAtom{Traits, DFrgType, AFrgType, BFrgType}) where {Traits, DFrgType, AFrgType, BFrgType}
    return make_fragment_like(BFrgType, B)
end

@inline function make_fragment_C(C::CuTeArray, m::MMAAtom{Traits, DFrgType, AFrgType, BFrgType}) where {Traits, DFrgType, AFrgType, BFrgType, CFrgType}
    @assert rank(C) ≥ static(3)
    @assert size(C, 1) == size(m.traits.Clayout, 1)
    return make_fragment_like(CFrgType, shape(C))
end
