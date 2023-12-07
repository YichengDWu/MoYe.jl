function Base.copyto!(copy_atom::CopyAtom{Traits, T, OP}, dst::StaticOwningArray,
                      src::SharedArray) where {Traits, T, OP <: AbstractLdMatrix}
    @inline
    buffer = ManualMemory.preserve_buffer(dst)
    GC.@preserve buffer begin copyto!(copy_atom, StrideArraysCore.maybe_ptr_array(dst), src) end
    return dst
end
