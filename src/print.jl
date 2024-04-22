function Base.show(io::IO, l::Layout)
    return print(io, shape(l), ":", stride(l))
end

@inline Base.ndigits(@nospecialize x::StaticInt{N}) where {N} = ndigits(N)

function print_layout(layout::Layout{2})
    idx_width = ndigits(cosize(layout)) + 2
    delim = "+-----------------"

    print(layout)
    print("\n")

    # Column indices
    print("    ")
    for n in 1:size(layout, 2)
        formatted_number = lpad(n, idx_width - 2)
        print("  ", formatted_number, " ")
    end
    println()

    # Print out A m-by-n
    for m in 1:size(layout, 1)
        # Header
        print("    ")
        for n in 1:size(layout, 2)
            print(view(delim, 1:(idx_width + 1)))
        end
        println("+")
        # Values
        print(lpad(m, 2), "  ")  # Row indices
        for n in 1:size(layout, 2)
            formatted_number = lpad(Int(layout(m, n)), idx_width - 2)
            print("| ", formatted_number, " ")
        end
        println("|")
    end
    # Footer
    print("    ")
    for n in 1:size(layout, 2)
        print(view(delim, 1:(idx_width + 1)))
    end
    return println("+")
end

function print_typst(
    A::Layout, TA::Layout, 
    B::Layout, TB::Layout,
    C::Layout, TC::Layout
)
    @assert size(A, 1) == size(C, 1)
    @assert size(B, 1) == size(C, 2)
    @assert size(A, 2) == size(B, 2)

    sA = known(size(TA))
    sB = known(size(TB))
    sC = known(size(TC))

    height = (size(A, 1) + size(A, 2) + 2 + 3) * 2.34
    width = (size(B, 1) + size(B, 2) + 3) * 2.34

    # build table C
    table_C_cells = Vector{String}()
    for m in 1:size(C, 1)
        push!(table_C_cells, "fill_cell(1),")
        for n in 1:size(C, 2)
            val_idx, thr_id = fldmod1(C(m, n), sC)
            thr_idx = TC(thr_id)
            push!(table_C_cells, "cell($thr_idx, $val_idx),")
        end
        push!(table_C_cells, "\n")
    end

    table_A_cells = Vector{String}()
    for m in 1:size(A, 1)
        push!(table_A_cells, "left_cell($m),")
        for k in 1:size(A, 2)
            val_idx, thr_id = fldmod1(A(m, k), sA)
            thr_idx = TA(thr_id)
            push!(table_A_cells, "cell($thr_idx, $val_idx),")
        end
        push!(table_A_cells, "\n")
    end

    table_B_cells = Vector{String}()    
    for k in 1:size(B, 2)
        push!(table_B_cells, "left_cell($k),")
        for n in 1:size(B, 1)
            val_idx, thr_id = fldmod1(B(n, k), sB)
            thr_idx = TB(thr_id)
            push!(table_B_cells, "cell($thr_idx, $val_idx),")
        end
        push!(table_B_cells, "\n")
    end


    typst_string = """
#set par(leading: 0.25em)
#set page(width: $(width)em, height: $(height)em, margin: 0pt)

#let color_map = (
    rgb(175, 175, 255),
    rgb(175, 255, 175),
    rgb(255, 255, 175),
    rgb(255, 175, 175),
    rgb(210, 210, 255),
    rgb(210, 255, 210),
    rgb(255, 210, 210),
    aqua,
    teal,
    red.transparentize(50%),
    yellow.transparentize(50%),
    lime.transparentize(50%),
    fuchsia.transparentize(50%),
    color.mix((aqua, 10%), (red, 50%)).transparentize(10%),
    orange.transparentize(30%),
    purple.transparentize(50%),  
    maroon.transparentize(70%),
    rgb(87, 127, 230).transparentize(30%),
    cmyk(27%, 0%, 3%, 5%),
    color.hsl(30deg, 50%, 60%).transparentize(30%),
    color.mix((red, 70%), (blue, 30%)).transparentize(30%),
    color.mix((red, 70%), (fuchsia, 30%)).transparentize(30%),
    color.mix((aqua, 70%), (fuchsia, 30%)).transparentize(10%),
    color.mix((purple, 70%), (blue, 30%)).transparentize(50%),
    color.mix((red, 30%), (fuchsia, 70%)).transparentize(50%),
    color.mix((eastern, 30%), (fuchsia, 70%)).transparentize(30%),
    color.mix((green, 50%), (olive, 50%)).transparentize(30%),
    color.mix((blue, 50%), (purple, 50%)).transparentize(30%),
    color.mix((yellow, 50%), (purple, 50%)).transparentize(30%),
    color.mix((orange, 50%), (purple, 50%)).transparentize(30%),
    color.mix((teal, 70%), (blue, 30%)).transparentize(30%),
    color.mix((aqua, 60%), (lime, 40%)).transparentize(30%),
)

#let cell(thr_idx, val_idx) = table.cell(fill: color_map.at((thr_idx.bit-and(31))))[#block(width: 1.5em, height: 1.5em)[T#(thr_idx) \\ V#(val_idx)]]
#let top_cell(i) = table.cell(stroke:none, align: horizon)[#(if i == 0 {block(width: 1.5em, height: 1.5em)[]} else {block(width: 1.5em, height: 1.5em)[#i]})]
#let left_cell(i) = table.cell(x: 0, y:i, stroke:none, align: horizon)[#block(width: 1.5em, height: 1.5em)[#i]]
#let fill_cell(i) = table.cell(stroke:none)[#block(width: 1.5em, height: 1.5em)[]]

#let table_C = table(
    rows: $(known(size(C, 1))),
    columns: $(known(size(C, 2)+One())),
    align: center,
    table.header(..range($(known(size(C, 2)+One()))).map(fill_cell)),
    $(table_C_cells...)
)

#let table_A = table(
    columns: $(known(size(A, 2)+One())),
    rows: $(known(size(A, 1))),
    align: center,
    table.header(..range($(known(size(A, 2)+One()))).map(top_cell)),
    $(table_A_cells...)
)


#let table_B = table(
    columns: $(known(size(B, 1)+One())),
    rows: $(known(size(B, 2))),
    align: center,
    table.header(..range($(known(size(B, 1)+One()))).map(top_cell)),
    $(table_B_cells...)
)


#grid(
    columns: 2,
    rows: 2,
    gutter: 0pt,
    [], table_B,
    table_A, table_C
)
    """
    print(typst_string)
end


"""
    print_typst(::MMAAtom)

Print the layout of the A, B, and C matrices in a typst format.
Go to https://typst.app and paste the output to visualize the layout.

## Example
```
julia> tiled_mma = make_tiled_mma(MMAOP_8x8x4_F32F16F16F32_NT(), @Layout((2,2), (2,1)), (@Layout((4,4,2), (1,8,4)), _32, _4))
TiledMMA
  ThrLayoutVMNK: ((_4, _2), _2, _2, _1):((_1, _16), _8, _4, _0)
  PermutationMNK: ((_4, _4, _2):(_1, _8, _4), _32, _4)
MMAAtom
  Thread ID: (_4, _2):(_1, _16)
  Layout_A_TV: ((_4, _2), _4):((_8, _4), _1)
  Layout_B_TV: ((_4, _2), _4):((_8, _4), _1)
  Layout_C_TV: ((_2, _2, _2), (_2, _2, _2)):((_1, _16, _4), (_8, _2, _32))


julia> print_typst(tiled_mma)
```

It will print the following image:
![](../assets/tiled_mma.png)
"""
print_typst(m::MMAAtom) = print_typst(make_tiled_mma(m))
function print_typst(tiled_mma::TiledMMA)
    A, TA = get_layoutA_MK(tiled_mma)
    B, TB = get_layoutB_NK(tiled_mma)
    C, TC = get_layoutC_MN(tiled_mma)

    print_typst(A, TA, B, TB, C, TC)
end