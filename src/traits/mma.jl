struct MMATraits{M <: MMAOP, DElType, AElType, BElType, CElType, DFrgType, AFrgType,
                 BFrgType, CFrgType, S, T, A, B, C}
    mnk::S
    threadid::T
    Alayout::A
    Blayout::B
    Clayout::C
end

export MMATraits

# For Hooper, FrgType and ElTypeare not the same
@inline function MMATraits{M, DElType, AElType, BElType, CElType}(mnk, threadid, Alayout,
                                                                  Blayout,
                                                                  Clayout) where {M,
                                                                                  DElType,
                                                                                  AElType,
                                                                                  BElType,
                                                                                  CElType}
    return MMATraits{M, DElType, AElType, BElType, CElType, DElType, AElType, BElType,
                     CElType, typeof(mnk), typeof(threadid), typeof(Alayout),
                     typeof(Blayout), typeof(Clayout)}(mnk, threadid, Alayout, Blayout,
                                                       Clayout)
end

@inline function fragtype_d(::MMATraits{M, DElType, AElType, BElType, CElType, DFrgType,
                                        AFrgType, BFrgType, CFrgType}) where {M, DElType,
                                                                              AElType,
                                                                              BElType,
                                                                              CElType,
                                                                              DFrgType,
                                                                              AFrgType,
                                                                              BFrgType,
                                                                              CFrgType}
    return DFrgType
end
@inline function fragtype_a(::MMATraits{M, DElType, AElType, BElType, CElType, DFrgType,
                                        AFrgType, BFrgType, CFrgType}) where {M, DElType,
                                                                              AElType,
                                                                              BElType,
                                                                              CElType,
                                                                              DFrgType,
                                                                              AFrgType,
                                                                              BFrgType,
                                                                              CFrgType}
    return AFrgType
end
@inline function fragtype_b(::MMATraits{M, DElType, AElType, BElType, CElType, DFrgType,
                                        AFrgType, BFrgType, CFrgType}) where {M, DElType,
                                                                              AElType,
                                                                              BElType,
                                                                              CElType,
                                                                              DFrgType,
                                                                              AFrgType,
                                                                              BFrgType,
                                                                              CFrgType}
    return BFrgType
end
@inline function fragtype_c(::MMATraits{M, DElType, AElType, BElType, CElType, DFrgType,
                                        AFrgType, BFrgType, CFrgType}) where {M, DElType,
                                                                              AElType,
                                                                              BElType,
                                                                              CElType,
                                                                              DFrgType,
                                                                              AFrgType,
                                                                              BFrgType,
                                                                              CFrgType}
    return CFrgType
end

function MMATraits{UniversalFMA{D,A,B,C}}() where {D,A,B,C}
    mnk = (static(1), static(1), static(1))
    threadid = @Layout 1
    Alayout = @Layout (1, 1)
    Blayout = @Layout (1, 1)
    Clayout = @Layout (1, 1)
    return MMATraits{UniversalFMA{D,A,B,C}, D, A, B, C}(mnk, threadid, Alayout, Blayout,
                                                        Clayout)
end

function mmaop_to_layoutargs(s::String)
    split_str = split(s, "_")

    num_pattern = r"\d+"
    letter_pattern = r"[A-Z]+\d+"

    num_matches = collect(eachmatch(num_pattern, split_str[2]))
    letter_matches = collect(eachmatch(letter_pattern, split_str[3]))

    mnk = tuple((static(parse(Int, m.match)) for m in num_matches)...)
    eltypes = tuple((ptx_to_jl[lowercase(m.match)] for m in letter_matches)...)

    return mnk, eltypes, Val(Symbol(split_str[4]))
end

function _get_layouts(::Tuple{StaticInt{16}, StaticInt{8}, StaticInt{8}},
                      AElType::Type{<:Union{Float16, BFloat16}},
                      CEltyp::Type{<:Union{Float16, Float32}}, ::Val{:TN})
    threadid = @Layout(32)
    Alayout = @Layout ((4, 8), (2, 2)) ((32, 1), (16, 8))
    Blayout = @Layout ((4, 8), 2) ((16, 1), 8)
    Clayout = Alayout

    return threadid, Alayout, Blayout, Clayout
end

function _get_layouts(::Tuple{StaticInt{16}, StaticInt{8}, StaticInt{16}},
                      AElType::Type{<:Union{Float16, BFloat16}},
                      CElType::Type{<:Union{Float16, Float32}}, ::Val{:TN})
    threadid = @Layout(32)
    Alayout = @Layout ((4, 8), (2, 2, 2)) ((32, 1), (16, 8, 128))
    Blayout = @Layout ((4, 8), (2, 2)) ((16, 1), (8, 64))
    Clayout = @Layout ((4, 8), (2, 2)) ((32, 1), (16, 8))

    return threadid, Alayout, Blayout, Clayout
end

function make_mmatraits(mmaops)
    for mmaop in mmaops
        mnk, eltypes, major = mmaop_to_layoutargs(mmaop)
        DElType, AElType, BElType, CElType= eltypes
        layouts = _get_layouts(mnk, AElType, CElType, major)
        @eval @inline function MMATraits{$(Symbol(mmaop))}()
            return MMATraits{$(Symbol(mmaop)), $DElType, $AElType, $BElType, $CElType}($mnk,
                                                                                       $(layouts...))
        end
    end
end

# 16x8x8
make_mmatraits([
                   "MMAOP_16x8x8_F16F16F16F16_TN",
                   "MMAOP_16x8x8_F32F16F16F32_TN",
                   #"MMAOP_16x8x8_F32TF32TF32F32_TN", # TODO: add support for TF32
                   "MMAOP_16x8x8_F32BF16BF16F32_TN",
               ])

# 16x8x16
make_mmatraits([
                   "MMAOP_16x8x16_F16F16F16F16_TN",
                   "MMAOP_16x8x16_F32F16F16F32_TN",
                   "MMAOP_16x8x16_F32BF16BF16F32_TN",
               ])

# 8x8x4
