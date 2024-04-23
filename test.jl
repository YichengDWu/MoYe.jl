using CUDA, MoYe
function kernel(m, n, C, ldc)
    mC = MoYeArray(C, make_layout((m,n), (_1, ldc)))

    CtaShape = (_128, _128, _8)
    cta_coord = (blockIdx().x, blockIdx().y, :)

    ctaC = @tile mC CtaShape cta_coord (_1, _1, :)
    acc = make_fragment_like(Float32, make_layout((_8,_8)))

    for i in 1:size(acc)
        @inbounds acc[i] = Float32(i) * 0.01
    end

    threadC = @tile ctaC (8, 8) (1, 1)

    copy!(threadC, acc)
end


function main()
    dev_buffer = CUDA.zeros(Float32, 9, 9)
    @cuda threads=(1,1) blocks=(1,1) kernel(9, 9, dev_buffer, 9)
    CUDA.synchronize()
    println(dev_buffer)
end

function kernel()
    weird = right_reverse(make_layout(_2, _1))
    @cuprint "weird(_2)"
   
   
    return
end

@cuda threads=1 blocks=1 kernel()


MemShapeN = 8
MemShapeK = 3
TileN = 3
TileK = 2

MemShapeNRoundUp = cld(MemShapeN, TileN) * TileN
MemShapeKRoundUp = cld(MemShapeK, TileK) * TileK