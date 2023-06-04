using MoYe, Test, CUDA
using Core: LLVMPtr

@testset "Universal Copy" begin
    @testset "Explicit Vectorized Copy" begin
        @testset "Sequential Copy" begin
            # single thread
            a = [Int32(i) for i in 1:8]
            pa = reinterpret(LLVMPtr{Int32, AS.Generic}, pointer(a))
            x = MoYeArray(pa, @Layout((4,2)))

            b = [Int32(i) for i in 9:16]
            pb = reinterpret(LLVMPtr{Int32, AS.Generic}, pointer(b))
            y = MoYeArray(pb, @Layout((4,2)))

            GC.@preserve a b begin
                MoYe.copyto_vec!(y, x, Int128)
                @test y == x
            end
        end

        @testset "Parallized Copy" begin
            function parallelized_copy(a, b, thread_layout)
                for i in 1:size(thread_layout) # on gpu this is parallelized instead of sequential
                    thread_tile_a = @parallelize a thread_layout i
                    thread_tile_b = @parallelize b thread_layout i
                    display(thread_tile_a)
                    MoYe.copyto_vec!(thread_tile_b, thread_tile_a, Int32) # no vectorization here
                end
            end

            a_data = [Int32(i) for i in 1:6*6]
            b_data = [Int32(i) for i in 6*6+1:6*6*2]

            GC.@preserve a_data b_data begin
                thread_layout = @Layout((2,3))
                layout = @Layout((6,6))

                pa = pointer(a_data)
                pa = reinterpret(LLVMPtr{Int32, AS.Generic}, pa)
                a = MoYeArray(pa, layout)

                pb = pointer(b_data)
                pb = reinterpret(LLVMPtr{Int32, AS.Generic}, pb)
                b = MoYeArray(pb, layout)
                parallelized_copy(a, b, thread_layout)
            end

            @test b == a
        end
    end

    @testset "Auto-vectorized Copy" begin
        @testset "Sequential Copy" begin
            a = [Int32(i) for i in 1:8]
            pa = reinterpret(LLVMPtr{Int32, AS.Generic}, pointer(a))
            x = MoYeArray(pa, @Layout((4,2)))

            b = [Int32(i) for i in 9:16]
            pb = reinterpret(LLVMPtr{Int32, AS.Generic}, pointer(b))
            y = MoYeArray(pb, @Layout((4,2)))

            GC.@preserve b a begin
                copyto!(y, x) # should recast to UInt128
                @test y == x
            end

            @test_deprecated cucopyto!(y, x)
        end

        @testset "Parallized Copy" begin
            function parallelized_copy(a, b, thread_layout)
                for i in 1:size(thread_layout) # on gpu this is parallelized instead of sequential
                    thread_tile_a = @parallelize a thread_layout i
                    thread_tile_b = @parallelize b thread_layout i
                    display(thread_tile_a)
                    copyto!(thread_tile_b, thread_tile_a)
                end
            end

            a_data = [Int32(i) for i in 1:6*6]
            b_data = [Int32(i) for i in 6*6+1:6*6*2]

            GC.@preserve a_data b_data begin
                thread_layout = @Layout((2,3))
                layout = @Layout((6,6))

                pa = pointer(a_data)
                pa = reinterpret(LLVMPtr{Int32, AS.Generic}, pa)
                a = MoYeArray(pa, layout)

                pb = pointer(b_data)
                pb = reinterpret(LLVMPtr{Int32, AS.Generic}, pb)
                b = MoYeArray(pb, layout)
                parallelized_copy(a, b, thread_layout)
            end

            @test b == a
        end
    end
end

@testset "Tiled Copy" begin
    @testset "UniversalCopy" begin
        function tiled_copy_kernel(g_in, g_out, tiled_copy, smem_layout)
            t_g_in = MoYeArray(pointer(g_in), smem_layout)
            t_g_out = MoYeArray(pointer(g_out), smem_layout)
            t_smem=MoYeArray{UInt16}(undef, smem_layout)

            for tid in 1:32
                for i in tid:size(tiled_copy):size(t_smem.layout)
                    t_smem[i] = t_g_in[i]
                end
            end

            for tid in 1:size(tiled_copy)
                thr_copy = get_thread_slice(tiled_copy, tid)
                tXsX = partition_S(thr_copy, t_smem)
                tXgX = partition_D(thr_copy, t_g_out)
                tXrX = MoYeArray{UInt16}(undef, tXgX.layout.shape)
                copyto!(tiled_copy, tXrX, tXsX)
                copyto!(tXgX, tXrX)
            end
        end
        @testset "32 x 32" begin
            g_in = [UInt16(i) for i in 1:32*32]
            g_out = zeros(UInt16, 32*32)
            smem_layout = @Layout (32,32) (1,32)
            tiled_copy = make_tiled_copy(MoYe.CopyAtom{MoYe.UniversalCopy{UInt16, UInt16}, UInt16}(),
                                         @Layout((16,2)), @Layout((2,4)))

            tiled_copy_kernel(g_in, g_out, tiled_copy, smem_layout)
            @test g_out == g_in
        end

        @testset "32 x 8" begin
            g_in = [UInt16(i) for i in 1:32*8]
            g_out = zeros(UInt16, 32*8)
            smem_layout = @Layout (32, (2, 4)) (2, (1, 64))
            tiled_copy = make_tiled_copy(MoYe.CopyAtom{MoYe.UniversalCopy{UInt16, UInt16}, UInt16}(),
                                         @Layout((32,1)), @Layout((1,8)))
            tiled_copy_kernel(g_in, g_out, tiled_copy, smem_layout)
            @test g_out == g_in
        end
    end
end
