function make_mmatraits_sm70(mmaops)
    for mmaop in mmaops
        @eval @inline thr_id(::MMATraits{$mmaop}) = @Layout (4, 2) (1, 16)
        
        mmaop_ins = @eval $mmaop()
        ALayout = alignment_a(mmaop_ins) == :row ? @Layout((8, 4), (1, 8)) : @Layout(((4, 2), 4), ((8, 4), 1))
        BLayout = alignment_b(mmaop_ins) == :col ? @Layout((8, 4), (1, 8)) : @Layout(((4, 2), 4), ((8, 4), 1))
        CLayout = valtype_c(mmaop_ins) == Float16 ? @Layout((8, 8), (1, 8)) : @Layout(((2,2,2), (2,2,2)), ((1,16,4), (8,2,32)))
    
        @eval layout_a(::MMATraits{$mmaop}) = $ALayout
        @eval layout_b(::MMATraits{$mmaop}) = $BLayout
        @eval layout_c(::MMATraits{$mmaop}) = $CLayout
    end
end

make_mmatraits_sm70((
    :MMAOP_8x8x4_F32F16F16F32_TN,
    :MMAOP_8x8x4_F32F16F16F32_NT,
    :MMAOP_8x8x4_F32F16F16F32_TT,
    :MMAOP_8x8x4_F32F16F16F32_NN,
    :MMAOP_8x8x4_F16F16F16F16_TN,
    :MMAOP_8x8x4_F16F16F16F16_NT,
    :MMAOP_8x8x4_F16F16F16F16_TT,
    :MMAOP_8x8x4_F16F16F16F16_NN,
))
