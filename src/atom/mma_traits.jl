struct MMATraits{M <: MMAOP, DEltype, AEltype, BEltype, CEltype, DFrgtype, AFrgtype,
                 BFrgtype, CFrgtype, S, T, A, B, C}
    mnk::S
    threadid::T
    Alayout::A
    Blayout::B
    Clayout::C
end

# For Hooper, Frgtype and Eltype are not the same
@inline function MMATraits{M, DEltype, AEltype, BEltype, CEltype}(mnk, threadid, Alayout,
                                                                  Blayout,
                                                                  Clayout) where {M,
                                                                                  DEltype,
                                                                                  AEltype,
                                                                                  BEltype,
                                                                                  CEltype}
    return MMATraits{M, DEltype, AEltype, BEltype, CEltype, DEltype, AEltype, BEltype,
                     CEltype, typeof(mnk), typeof(threadid), typeof(Alayout),
                     typeof(Blayout), typeof(Clayout)}(mnk, threadid, Alayout, Blayout,
                                                       Clayout)
end

export MMATraits

@inline function fragtype_d(::MMATraits{M, DEltype, AEltype, BEltype, CEltype, DFrgtype,
                                        AFrgtype, BFrgtype, CFrgtype}) where {M, DEltype,
                                                                              AEltype,
                                                                              BEltype,
                                                                              CEltype,
                                                                              DFrgtype,
                                                                              AFrgtype,
                                                                              BFrgtype,
                                                                              CFrgtype}
    return DFrgtype
end
@inline function fragtype_a(::MMATraits{M, DEltype, AEltype, BEltype, CEltype, DFrgtype,
                                        AFrgtype, BFrgtype, CFrgtype}) where {M, DEltype,
                                                                              AEltype,
                                                                              BEltype,
                                                                              CEltype,
                                                                              DFrgtype,
                                                                              AFrgtype,
                                                                              BFrgtype,
                                                                              CFrgtype}
    return AFrgtype
end
@inline function fragtype_b(::MMATraits{M, DEltype, AEltype, BEltype, CEltype, DFrgtype,
                                        AFrgtype, BFrgtype, CFrgtype}) where {M, DEltype,
                                                                              AEltype,
                                                                              BEltype,
                                                                              CEltype,
                                                                              DFrgtype,
                                                                              AFrgtype,
                                                                              BFrgtype,
                                                                              CFrgtype}
    return BFrgtype
end
@inline function fragtype_c(::MMATraits{M, DEltype, AEltype, BEltype, CEltype, DFrgtype,
                                        AFrgtype, BFrgtype, CFrgtype}) where {M, DEltype,
                                                                              AEltype,
                                                                              BEltype,
                                                                              CEltype,
                                                                              DFrgtype,
                                                                              AFrgtype,
                                                                              BFrgtype,
                                                                              CFrgtype}
    return CFrgtype
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
                      AEltype::Type{<:Union{Float16, BFloat16}},
                      CEltyp::Type{<:Union{Float16, Float32}}, ::Val{:TN})
    threadid = @Layout(32)
    Alayout = @Layout ((4, 8), (2, 2)) ((32, 1), (16, 8))
    Blayout = @Layout ((4, 8), 2) ((16, 1), 8)
    Clayout = Alayout

    return threadid, Alayout, Blayout, Clayout
end

function _get_layouts(::Tuple{StaticInt{16}, StaticInt{8}, StaticInt{16}},
                      AEltype::Type{<:Union{Float16, BFloat16}},
                      CEltype::Type{<:Union{Float16, Float32}}, ::Val{:TN})
    threadid = @Layout(32)
    Alayout = @Layout ((4, 8), (2, 2, 2)) ((32, 1), (16, 8, 128))
    Blayout = @Layout ((4, 8), (2, 2)) ((16, 1), (8, 64))
    Clayout = @Layout ((4, 8), (2, 2)) ((32, 1), (16, 8))

    return threadid, Alayout, Blayout, Clayout
end

function make_mmatraits(mmaops)
    for mmaop in mmaops
        mnk, eltypes, major = mmaop_to_layoutargs(mmaop)
        DEltype, AEltype, BEltype, CEltype = eltypes
        layouts = _get_layouts(mnk, AEltype, CEltype, major)
        @eval @inline function MMATraits{$(Symbol(mmaop))}()
            return MMATraits{$(Symbol(mmaop)), $DEltype, $AEltype, $BEltype, $CEltype}($mnk,
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
