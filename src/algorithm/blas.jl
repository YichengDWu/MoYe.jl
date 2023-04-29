@inline function axpby!(a::Number, X::CuTeArray{T}, b::Number, Y::CuTeArray{T}) where {T}
    x, y = ManualMemory.preserve_buffer(X), ManualMemory.preserve_buffer(Y)
    vx, vy = ViewEngine(engine(X)), ViewEngine(engine(Y))
    GC.@preserve x y begin
        @unroll for i in eachindex(vx)
            vy[i] = iszero(b) ? a * vx[i] : a * vx[i] + b * vy[i]
        end
    end
end
