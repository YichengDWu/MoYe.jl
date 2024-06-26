# MoYeArray

[`MoYeArray`](@ref) befinits from the [`Layout`](@ref) to be able to create
all kinds of special arrays. For example, we can create a `FillArray`-like array

```@repl array
using MoYe
MoYeArray{Float64}(one, @Layout((3,4), (0, 0)))
ans.engine
```
As you can see the array only contains one element. Under the hood, the physical length
of that array is calculated by [`cosize`](@ref):
```@repl array
cosize(@Layout((3,4), (0, 0)))
```

The underlying implementation of MoYeArray determines that linear indexing is actually periodic.
```@repl array
function f()
    B = MoYeArray([1,2,3], @Layout((3,), (1,)))
    @show @inbounds B[4], B[5], B[6], B[7]
end
f();
```

We can also easily create a so-called `BlockArray`:
```@repl array
data = collect(1:48);
B=MoYeArray(data, @Layout(((2,3), (2,4)), ((1, 16), (2, 4))))
```
Here we created a 2x3 block array with 2x4 blocks. The first mode is the index
of the block, the second mode is the index within the block.


## Slicing 

It is **required** to use the  syntax `view(a, ids...) or `@view a[ids...]`, depenting on your tast. 
```@repl array
data = [i for i in 1:164];
a = MoYeArray(data, ((_3, 2), (2, _5, _2)), ((4,1), (_2, 13, 100)))
b = @view a[2, :]
```
