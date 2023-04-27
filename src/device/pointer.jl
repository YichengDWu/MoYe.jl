@inline isgmem(::CuTeArray{T, N, ViewEngine{T, LLVMPtr{M, AS.Global}}}) where {T, N, M} = true
@inline isgmem(::CuTeArray) = false

@inline issmem(::CuTeArray{T, N, ViewEngine{T, LLVMPtr{M, AS.Shared}}}) where {T, N, M} = true
@inline issmem(::CuTeArray) = false

@inline isrmem(::CuTeArray) = !isgmem(x) && !issmem(x)
