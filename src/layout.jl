struct Layout{N, Shape, Stride}
    shape::Shape
    stride::Stride

    Layout(shape::IntTuple, stride::IntTuple) = new{rank(shape), typeof(shape), typeof(stride)}(shape, stride)
    Layout(shape::Int, stride::Int) = new{1, typeof(shape), typeof(stride)}(shape, stride)
end

function (l::Layout)(coord::IntTuple)
    coord_to_index(coord, l.shape, l.stride)
end
