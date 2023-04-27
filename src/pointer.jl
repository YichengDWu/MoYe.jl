@inline isgmem(::CuTeArray{T, N, ViewEngine{T, LLVMPtr{M, AS.Global}}}) where {T, N, M} = true
@inline isgmem(::CuTeArray) = false

@inline issmem(::CuTeArray{T, N, ViewEngine{T, LLVMPtr{M, AS.Shared}}}) where {T, N, M} = true
@inline issmem(::CuTeArray) = false

@inline isrmem(::CuTeArray) = !isgmem(x) && !issmem(x)

@inline recast(::Type{T}, ptr::LLVMPtr{S, AS}) where {T, S, AS} = LLVM.Interop.addrspacecast(LLVMPtr{T, AS}, ptr)
@inline recast(::Type{T}, ptr::Ptr) where {T} = reinterpret(Ptr{T}, ptr)
