@inline Base.ndigits(@nospecialize ::StaticInt{N}) where {N} = ndigits(N)

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
