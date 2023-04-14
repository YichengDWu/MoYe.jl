function print_layout(layout::Layout{2})
    idx_width = ndigits(cosize(layout)) + 2
    delim = "+-----------------"

    print(layout); print("\n")

    # Column indices
    print("    ")
    for n = 1:size(layout, 2)
        formatted_number = lpad(n, idx_width - 2)
        print("  ", formatted_number, " ")
    end
    println()

    # Print out A m-by-n
    for m = 1:size(layout, 1)
        # Header
        print("    ")
        for n = 1:size(layout, 2)
            print(view(delim, 1:idx_width + 1))
        end
        println("+")
        # Values
        print(lpad(m, 2), "  ")  # Row indices
        for n = 1:size(layout, 2)
            formatted_number = lpad(Int(layout(m, n)), idx_width - 2)
            print("| ", formatted_number, " ")
        end
        println("|")
    end
    # Footer
    print("    ")
    for n = 1:size(layout, 2)
        print(view(delim, 1:idx_width + 1))
    end
    println("+")
end
