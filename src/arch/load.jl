abstract type LdMatrix{DRegisters, SRegisters} end

function Base.getproperty(obj::LdMatrix{DRegisters, SRegisters},
                          sym::Symbol) where {DRegisters, SRegisters}
    if sym === :DRegisters
        return DRegisters
    elseif sym === :SRegisters
        return SRegisters
    else
        return getfield(obj,sym)
    end
end
