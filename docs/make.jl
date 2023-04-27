using Shambles
using Documenter

DocMeta.setdocmeta!(Shambles, :DocTestSetup, :(using Shambles); recursive=true)

makedocs(; modules=[Shambles],
         authors="MilkshakeForReal <yicheng.wu@ucalgary.ca> and contributors",
         repo="https://github.com/YichengDWu/Shambles.jl/blob/{commit}{path}#{line}",
         sitename="Shambles.jl",
         format=Documenter.HTML(; prettyurls=get(ENV, "CI", "false") == "true",
                                canonical="https://YichengDWu.github.io/Shambles.jl",
                                edit_link="main", assets=String[]),
         pages=[
            "Home" => "index.md",
            "API Reference" => [
                    "Layout" => "api/layout.md",
                    "CuTeArray" => "api/array.md",
                    "Tiling" => "api/tiling.md",
                ],
         ])

deploydocs(; repo="github.com/YichengDWu/Shambles.jl", push_preview=true, devbranch="main")
