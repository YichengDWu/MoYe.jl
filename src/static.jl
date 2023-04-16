macro static(expr)
    if expr.head == :call
        fname = expr.args[1]
        static_args = [:(static($(esc(arg)))) for arg in expr.args[2:end]]
        return Expr(:call, fname, static_args...)
    else
        error("Invalid expression passed to @static.")
    end
end
