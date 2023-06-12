function CopyTraits{LDSM_U32x1_N}()
    threadid = @Layout 32 # 32 threads per operation
    srclayout = @Layout ((8, 4), 128) ((128, 0), 1) # thr -> bit
    dstlayout = @Layout (32, 32) (32, 1)
    return CopyTraits{LDSM_U32x1_N}(threadid, srclayout, dstlayout, dstlayout)
end

function CopyTraits{LDSM_U32x2_N}()
    threadid = @Layout 32
    srclayout = @Layout ((16, 2), 128) ((128, 0), 1)
    dstlayout = @Layout (32, (32, 2)) (32, (1, 1024))
    return CopyTraits{LDSM_U32x2_N}(threadid, srclayout, dstlayout, dstlayout)
end

function CopyTraits{LDSM_U32x4_N}()
    threadid = @Layout 32
    srclayout = @Layout (32, 128) (128, 1)
    dstlayout = @Layout (32, (32, 4)) (32, (1, 1024))
    return CopyTraits{LDSM_U32x2_N}(threadid, srclayout, dstlayout, dstlayout)
end

function CopyTraits{LDSM_U16x2_T}()
    threadid = @Layout 32
    srclayout = @Layout ((8, 4), 128) ((128, 0), 1)
    dstlayout = @Layout ((4, 8), (16, 2)) ((256, 16), (1, 128))
    return CopyTraits{LDSM_U16x2_T}(threadid, srclayout, dstlayout, dstlayout)
end

function CopyTraits{LDSM_U16x4_T}()
    threadid = @Layout 32
    srclayout = @Layout ((16, 2), 128) ((128, 0), 1)
    dstlayout = @Layout ((4, 8), (16, 2, 2)) ((256, 16), (1, 128, 1024))
    return CopyTraits{LDSM_U16x4_T}(threadid, srclayout, dstlayout, dstlayout)
end

function CopyTraits{LDSM_U16x8_T}()
    threadid = @Layout 32
    srclayout = @Layout (32, 128) (128, 1)
    dstlayout = @Layout ((4, 8), (16, 2, 4)) ((256, 16), (1, 128, 1024))
    return CopyTraits{LDSM_U16x8_T}(threadid, srclayout, dstlayout, dstlayout)
end
