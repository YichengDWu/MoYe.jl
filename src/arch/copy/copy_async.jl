abstract type AbstractCopyOp_ASYNC{TS,TD} <: AbstractCopyOp{Registers{TS, 1}, Registers{TD, 1}} end
const CP_SYNC_ENABLED = @static if CUDA.functional() && capability(device()) >= v"8.0"
    true
else
    false
end

struct CPOP_ASYNC_CACHEALWAYS{TS, TD} <: AbstractCopyOp_ASYNC{TS,TD}
    @generated function CPOP_ASYNC_CACHEALWAYS{TS, TD}() where {TS, TD}
        @assert sizeof(TS) == sizeof(TD)
        @assert sizeof(TS) in (4, 8, 16) "Only 4, 8, 16 bytes are supported, got $(sizeof(TS))"
        return :($(new{TS, TD}()))
    end
end

@inline CPOP_ASYNC_CACHEALWAYS{S}() where {S} = CPOP_ASYNC_CACHEALWAYS{S,S}()

@generated function (::CPOP_ASYNC_CACHEALWAYS{TS, TD})(dst::LLVMPtr{TD, AS.Shared}, src::LLVMPtr{TS, AS.Global}) where {TD, TS}
    intr = "llvm.nvvm.cp.async.ca.shared.global.$(sizeof(TS))"
    return quote
        Base.@_inline_meta
        ccall($intr, llvmcall, Cvoid, (LLVMPtr{TD, AS.Shared}, LLVMPtr{TS, AS.Global}), dst, src)
    end
end

struct CPOP_ASYNC_CACHEGLOBAL{TS, TD} <: AbstractCopyOp_ASYNC{TS,TD}
    @generated function CPOP_ASYNC_CACHEGLOBAL{TS, TD}() where {TS, TD}
        @assert sizeof(TS) == sizeof(TD)
        @assert sizeof(TS) in (16, ) "Only 16 bytes are supported, got $(sizeof(TS))" # only 16 for LLVM 15
        return :($(new{TS, TD}()))
    end
end

@inline CPOP_ASYNC_CACHEGLOBAL{S}() where {S} = CPOP_ASYNC_CACHEGLOBAL{S,S}()

@generated function (::CPOP_ASYNC_CACHEGLOBAL{TS, TD})(dst::LLVMPtr{TD, AS.Shared}, src::LLVMPtr{TS, AS.Global}) where {TS, TD}
    intr = ".$(sizeof(TS))"
    return quote
        Base.@_inline_meta
        ccall($intr, llvmcall, Cvoid, (LLVMPtr{TD, AS.Shared}, LLVMPtr{TS, AS.Global}), dst, src)
    end
end

"""
    cp_async_wait(i::Int32)
    cp_async_wait()

`cp.async.wait.group` and `cp.async.wait.all`.
"""
@inline cp_async_wait(i::Int32) = ccall("llvm.nvvm.cp.async.wait.group", llvmcall, Cvoid, (Int32,), i)
@inline cp_async_wait() = ccall("llvm.nvvm.cp.async.wait.all", llvmcall, Cvoid, ())

"""
    cp_async_commit()

`cp.async.commit.group`.
"""
@inline cp_async_commit() = ccall("llvm.nvvm.cp.async.commit.group", llvmcall, Cvoid, ())
