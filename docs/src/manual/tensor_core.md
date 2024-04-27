# Tensor Cores

Tensor cores are specialized hardware accelerators designed to optimize matrix operations, which are crucial for deep learning and artificial intelligence algorithms.

Switching to tensor cores can be as simple as modifying just one line of code in the previous matmul function:

```julia
mma_C = make_tiled_mma(MMAOP_16x8x8_F32TF32TF32F32_TN(), 
                       @Layout((2,4,1)))
```

Let's explore what TiledMMA entails.
```julia
print_typst(mma_C)
```
![](../assets/tensorcore.svg)

At first glance, the diagram may seem complex, but the concept is straightforward: the threads collective load data from matrices A and B according to the specified layout. During the matrix multiply-accumulate (MMA) computation, data is internally shared among threads—a process that is not transparent to the user. Once the computation is complete, each thread stores the results as dictated by the layout of matrix C shown in the illustration.

Of course you can also choose other mma atoms. They just work.
```julia
mma_C = make_tiled_mma(MMAOP_16x8x8_F32F16F16F32_TN(), 
                                @Layout((2,4,1)))
```

## LDMatrix

