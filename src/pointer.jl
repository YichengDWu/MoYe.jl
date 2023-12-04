@inline isgmem(::MoYeArray{T, N, ViewEngine{T, LLVMPtr{M, AS.Global}}}) where {T, N, M} = true
@inline isgmem(::MoYeArray) = false

@inline issmem(::MoYeArray{T, N, ViewEngine{T, LLVMPtr{M, AS.Shared}}}) where {T, N, M} = true
@inline issmem(::MoYeArray) = false

@inline isrmem(x::MoYeArray) = !isgmem(x) && !issmem(x)

@inline recast(::Type{T}, ptr::LLVMPtr{S, AS}) where {T, S, AS} = LLVM.Interop.addrspacecast(LLVMPtr{T, AS}, ptr)
@inline recast(::Type{T}, ptr::Ptr) where {T} = reinterpret(Ptr{T}, ptr)

Base.:(-)(x::LLVMPtr, ::StaticInt{N}) where {N} = x - N
Base.:(-)(::StaticInt{N}, y::LLVMPtr) where {N} = y - N
Base.:(+)(x::LLVMPtr, ::StaticInt{N}) where {N} = x + N
Base.:(+)(::StaticInt{N}, y::LLVMPtr) where {N} = y + N
