using CuTe
using Documenter

DocMeta.setdocmeta!(CuTe, :DocTestSetup, :(using CuTe); recursive=true)

makedocs(;
    modules=[CuTe],
    authors="MilkshakeForReal <yicheng.wu@ucalgary.ca> and contributors",
    repo="https://github.com/YichengDWu/CuTe.jl/blob/{commit}{path}#{line}",
    sitename="CuTe.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://YichengDWu.github.io/CuTe.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/YichengDWu/CuTe.jl",
    devbranch="main",
)
