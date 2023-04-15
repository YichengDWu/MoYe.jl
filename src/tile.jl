const Tile = Tuple{Vararg{Union{Int, StaticInt, Layout}}}

#function tile(l1::Layout, l2::Layout)
#    return tiled_divide(l1, l2)
#end

#function tile(l1::Layout, l2::Layout, l3::Layout...)
#    return tiled_divide(l1, (l2, l3...))
#end
