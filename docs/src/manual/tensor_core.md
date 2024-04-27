# Tensor Cores

Tensor cores are specialized hardware accelerators designed to speed up matrix operations, which are fundamental to deep learning and artificial intelligence algorithms. 

To use tensor core, it only requires a line of code to change：

```julia
mma_C = make_tiled_mma(MMAOP_16x8x16_F32F16F16F32_TN(), 
                              @Layout((2,4,1)))
```

However, choosing such a `MMAAtom` poses requirements for the element types of the oprends and 
the layout of shared memory.

```
function matmul(A, B, C)
    bM = _128
    bN = _128
    bK = _16
    
    sA_layout = make_layout((bM, bK), (_1, bM + _1))
    sB_layout = make_layout((bN, bK), (_1, bN + _1))

    TA = Float16
    TB = Float16
    TC = Float32
	
    copy_A = make_tiled_copy(CopyAtom{CPOP_ASYNC_CACHEALWAYS{Int32}, TA}(),
                                    @Layout((32, 8)),
                                    @Layout((4, 1)))
    copy_B = make_tiled_copy(CopyAtom{CPOP_ASYNC_CACHEALWAYS{Int32}, TB}(),
                                        @Layout((32, 8)),
                                        @Layout((4, 1)))

    mma_C = make_tiled_mma(MMAOP_16x8x16_F32F16F16F32_TN(), 
                           @Layout((2,4,1)))

    threads = Int(size(mma_C))
    blocks = (cld(size(A, 1), bM), cld(size(B, 1), bN))

    @cuda threads=threads blocks=blocks matmul_kernel(A, sA_layout, copy_A,
                                                      B, sB_layout, copy_B,
                                                      C, mma_C)
end
```


The
```julia
mma_atom = MMAAtom{MMAOP_16x8x16_F32F16F16F32_TN}()
print_typst(mma_atom)
```
![matmuil](../assets/mma_atom.svg)



```julia
sA_buffer = collect(1:16*16) .|> Float16;
sB_buffer = collect(1:8*16) .|> Float16;

sA = MoYeArray(sA_buffer, @Layout((16,16)));
sB = MoYeArray(sB_buffer, @Layout((8,16)));

tiled_mma = make_tiled_mma(MMAAtom{MMAOP_16x8x16_F32F16F16F32_TN}());
thr_idx = 1;
thr_mma = get_slice(tiled_mma, thr_idx);

sA
tCsA = partition_A(thr_mma, sA);
@view tCsA[:,:,1]     

sB'
tCsB = partition_B(thr_mma, sB);
@view tCsB[:,:,1]            

tCrA = make_fragment_A(mma_C, tCsA)
copyto!(tCrA, tCsA)
```

(_1, _8, _8):(0, 16, 32768)
Layout{3, Tuple{Static.StaticInt{1}, Static.StaticInt{8}, Static.StaticInt{8}}, Tuple{Static.StaticInt{0}, Static.StaticInt{16}, Int64}}(...)

StaticLayout{3, Tuple{Static.StaticInt{1}, Static.StaticInt{8}, Static.StaticInt{8}}, Tuple{Static.StaticInt{0}, Static.StaticInt{1}, Static.StaticInt{8}}}(...)
(_1, _8, _8):(_0, _1, _8)


Layout{3, Tuple{Tuple{Static.StaticInt{2}, Static.StaticInt{2}}, Static.StaticInt{4}, Static.StaticInt{4}}, Tuple{Tuple{Int64, Static.StaticInt{8}}, Static.StaticInt{32}, Int64}}

((_2, _2), _4, _4):((2048, _8), _32, 65536)


StaticLayout{3, Tuple{Tuple{Static.StaticInt{2}, Static.StaticInt{2}}, Static.StaticInt{4}, Static.StaticInt{4}}, Tuple{Tuple{Static.StaticInt{1}, Static.StaticInt{2}}, Static.StaticInt{4}, Static.StaticInt{16}}}

((_2, _2), _4, _4):((_1, _2), _4, _16)