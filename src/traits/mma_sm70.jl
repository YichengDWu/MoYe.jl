function make_mmatraits_sm70(mmaops)
    for mmaop in mmaops
        mnk, eltypes, major = mmaop_to_layoutargs(mmaop)
        major = String(typeof(major).parameters[1])
        DElType, AElType, BElType, CElType = eltypes
        threadid = @Layout (4, 2) (1, 16)

        ALayout = if major[1] == 'T'
            @Layout (8, 4) (1, 8)
        elseif major[1] == 'N'
            @Layout ((4, 2), 4) ((8, 4), 1)
        else
            throw("Invalid major: $major")
        end

        BLayout = if major[2] == 'N'
            @Layout (8, 4) (1, 8)
        elseif major[2] == 'T'
            @Layout ((4, 2), 4) ((8, 4), 1)
        else
            throw("Invalid major: $major")
        end

        CLayout = if DElType == Float16
            @Layout (8, 8) (1, 8)
        else
            @Layout ((2,2,2), (2,2,2)) ((1,16,4), (8,2,32))
        end

        @eval function MMATraits{@eval($(Symbol(mmaop)))}()
            return MMATraits{$(Symbol(mmaop)), $DElType, $AElType, $BElType, $CElType}($mnk,
                                                                                       $threadid,
                                                                                       $ALayout,
                                                                                       $BLayout,
                                                                                       $CLayout)
        end
    end
end

make_mmatraits_sm70([
    "MMAOP_8x8x4_F64F64F64F64_TN",
    "MMAOP_8x8x4_F32F16F16F16_TN",
    "MMAOP_8x8x4_F32F16F16F16_NT",
    "MMAOP_8x8x4_F32F16F16F16_TT",
    "MMAOP_8x8x4_F32F16F16F16_NN",
    "MMAOP_8x8x4_F32F16F16F32_TN",
    "MMAOP_8x8x4_F32F16F16F32_NT",
    "MMAOP_8x8x4_F32F16F16F32_TT",
    "MMAOP_8x8x4_F32F16F16F32_NN",
    "MMAOP_8x8x4_F16F16F16F16_TN",
    "MMAOP_8x8x4_F16F16F16F16_NT",
    "MMAOP_8x8x4_F16F16F16F16_TT",
    "MMAOP_8x8x4_F16F16F16F16_NN",
])
