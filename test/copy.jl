using Shambles, Test
using Core: LLVMPtr

@testset "Universal Copy" begin
    @testset "Trivial Copy" begin
        # single thread
        a = [Int32(i) for i in 1:8]
        pa = reinterpret(LLVMPtr{Int32, AS.Generic}, pointer(a))
        x = CuTeArray(pa, @Layout((4,2)))

        b = [Int32(i) for i in 9:16]
        pb = reinterpret(LLVMPtr{Int32, AS.Generic}, pointer(b))
        y = CuTeArray(pb, @Layout((4,2)))

        Shambles.copyto_vec!(y, x, Int128)
        @test y == x
    end

    @testset "Parallized Copy" begin
        function parallelized_copy(a, b, thread_layout)
            for i in static(1):size(thread_layout) # on gpu this is parallelized instead of sequential
                thread_tile_a = local_partition(a, thread_layout, i)
                thread_tile_b = local_partition(b, thread_layout, i)
                display(thread_tile_a)
                Shambles.copyto_vec!(thread_tile_b, thread_tile_a, Int32) # no vectorization here
            end
        end

        a_data = [Int32(i) for i in 1:6*6]
        b_data = [Int32(i) for i in 6*6+1:6*6*2]

        GC.@preserve a_data b_data begin
            thread_layout = @Layout((2,3))
            layout = @Layout((6,6))

            pa = pointer(a_data)
            pa = reinterpret(LLVMPtr{Int32, AS.Generic}, pa)
            a = CuTeArray(pa, layout)

            pb = pointer(b_data)
            pb = reinterpret(LLVMPtr{Int32, AS.Generic}, pb)
            b = CuTeArray(pb, layout)
            parallelized_copy(a, b, thread_layout)
        end

        @test b == a
    end
end
