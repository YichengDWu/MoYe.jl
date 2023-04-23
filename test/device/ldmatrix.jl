
using Test, CuTe, CUDA
using CUDA: i32

if CUDA.functional()

    function kernel(op)
        A = CuStaticSharedArray(UInt16, (64,))
        a_frag = load(pointer(A) + (threadIdx().x - 1i32) << 4, op)
        @cushow Float32(sum(a_frag))
        return nothing
    end

    buf = IOBuffer()
    @device_code_ptx io = buf @cuda threads=32 kernel(LDSM_U32x2_N())
    asm = String(take!(copy(buf)))

end
