@inline function Base.fill!(x::CuTeArray{T, N, <:ArrayEngine}, val) where {T, N}
    b = ManualMemory.preserve_buffer(x)
    vb = ViewEngine(engine(x))
    GC.@preserve b begin
        @turbo for i in 1:length(vb)
            vb[i] = val
        end
    end
    return x
end

@inline function Base.sum(x::CuTeArray{T, N, <:ArrayEngine}) where {T, N}
    b = ManualMemory.preserve_buffer(x)
    vx = ViewEngine(engine(x))
    GC.@preserve b begin
        tmp = zero(T)
        @turbo for i in 1:length(vx)
            tmp += vx[i]
        end
        return tmp
    end
end

@inline function axpby!(a::Number, X::CuTeArray{T}, b::Number, Y::CuTeArray{T}) where {T}
    x, y = ManualMemory.preserve_buffer(X), ManualMemory.preserve_buffer(Y)
    vx, vy = ViewEngine(engine(X)), ViewEngine(engine(Y))
    GC.@preserve x y begin
        @turbo for i in eachindex(vx)
            vy[i] = iszero(b) ? a * vx[i] : a * vx[i] + b * vy[i]
        end
    end
end
