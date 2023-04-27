abstract type CPOP_ASYNC{TS,TD} <: CPOP{Registers{TS, 1}, Registers{TD, 1}} end

struct CPOP_ASYNC_CACHEALWAYS{TS, TD} <: CPOP_ASYNC{TS,TD}
    function CPOP_ASYNC_CACHEALWAYS{TS, TD}() where {TS, TD}
        @assert sizeof(TS) == sizeof(TD)
        return new{TS, TD}()
    end
end

struct CPOP_ASYNC_CACHEGLOBAL{TS, TD} <: CPOP_ASYNC{TS,TD}
    function CPOP_ASYNC_CACHEGLOBAL{TS, TD}() where {TS, TD}
        @assert sizeof(TS) == sizeof(TD)
        return new{TS, TD}()
    end
end

function _unsafe_copyto!(dst::LLVMPtr{TD, AS.Shared}, src::LLVMPtr{TS, AS.Global}, ::StaticInt{4}, ::CPOP_ASYNC_CACHEALWAYS{TS, TD}) where {TD, TS}
    @inline
    ccall("llvm.nvvm.cp.async.ca.shared.global.4", llvmcall, Cvoid,
          (LLVMPtr{TD, AS.Shared}, LLVMPtr{TS, AS.Global}), dst, src)
end
function _unsafe_copyto!(dst::LLVMPtr{TD, AS.Shared}, src::LLVMPtr{TS, AS.Global}, ::StaticInt{8}, ::CPOP_ASYNC_CACHEALWAYS{TS, TD}) where {TD, TS}
    @inline
    ccall("llvm.nvvm.cp.async.ca.shared.global.8", llvmcall, Cvoid,
          (LLVMPtr{TD, AS.Shared}, LLVMPtr{TS, AS.Global}), dst, src)
end
function _unsafe_copyto!(dst::LLVMPtr{TD, AS.Shared}, src::LLVMPtr{TS, AS.Global}, ::StaticInt{16}, ::CPOP_ASYNC_CACHEALWAYS{TS, TD}) where {TD, TS}
    @inline
    ccall("llvm.nvvm.cp.async.ca.shared.global.16", llvmcall, Cvoid,
          (LLVMPtr{TD, AS.Shared}, LLVMPtr{TS, AS.Global}), dst, src)
end
function _unsafe_copyto!(dst::LLVMPtr{TD, AS.Shared}, src::LLVMPtr{TS, AS.Global}, ::StaticInt{16}, ::CPOP_ASYNC_CACHEGLOBAL{TS, TD}) where {TS, TD}
    @inline
    ccall("llvm.nvvm.cp.async.cg.shared.global.16", llvmcall, Cvoid,
          (LLVMPtr{TD, AS.Shared}, LLVMPtr{TS, AS.Global}), dst, src)
end

function _unsafe_copyto!(dst::LLVMPtr{TD, AS.Shared}, src::LLVMPtr{TS, AS.Global}, cpop::CPOP_ASYNC{TS,TD}) where {TS, TD}
    @inline
    @assert sizeof(TS) == sizeof(TD)
    _unsafe_copyto!(dst, src, static(sizeof(TS)), cpop)
end

@inline cp_async_wait(i::Int32) = ccall("llvm.nvvm.cp.async.wait.group", llvmcall, Cvoid, (Int32,), i)
@inline cp_async_wait() = ccall("llvm.nvvm.cp.async.wait.all", llvmcall, Cvoid, ())
@inline cp_async_fence() = ccall("llvm.nvvm.cp.async.commit.group", llvmcall, Cvoid, ())
