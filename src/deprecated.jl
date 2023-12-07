function cucopyto!(dest, src)
    Base.depwarn("cucopy! is deprecated and will be removed in v1.1, use copy! instead", :cucopy!)
    return copy!(dest, src)
end
