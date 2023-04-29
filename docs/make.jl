using Moye
using Documenter

DocMeta.setdocmeta!(Moye, :DocTestSetup, :(using Moye); recursive=true)

makedocs(; modules=[Moye],
         authors="MilkshakeForReal <yicheng.wu@ucalgary.ca> and contributors",
         repo="https://github.com/YichengDWu/Moye.jl/blob/{commit}{path}#{line}",
         sitename="Moye.jl",
         format=Documenter.HTML(; prettyurls=get(ENV, "CI", "false") == "true",
                                canonical="https://YichengDWu.github.io/Moye.jl",
                                edit_link="main", assets=String[]),
         pages=[
            "Home" => "index.md",
            "Manual" => [
                "Layout" => "manual/layout.md",
                "Data Movement" => [
                    "Global Memory & Shared Memory" => "manual/datamovement/gs.md",
                ]
            ],
            "API Reference" => [
                    "Layout" => "api/layout.md",
                    "MoyeArray" => "api/array.md",
                    "Tiling" => "api/tiling.md",
                ],
         ])

deploydocs(; repo="github.com/YichengDWu/Moye.jl", push_preview=true, devbranch="main")
