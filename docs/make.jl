using MoYe
using Documenter

DocMeta.setdocmeta!(MoYe, :DocTestSetup, :(using MoYe); recursive=true)

makedocs(; modules=[MoYe],
         authors="MilkshakeForReal <yicheng.wu@ucalgary.ca> and contributors",
         repo="https://github.com/YichengDWu/MoYe.jl/blob/{commit}{path}#{line}",
         sitename="MoYe.jl",
         format=Documenter.HTML(; prettyurls=get(ENV, "CI", "false") == "true",
                                canonical="https://YichengDWu.github.io/MoYe.jl",
                                edit_link="main", assets=String[]),
         pages=[
            "Home" => "index.md",
            "Manual" => [
                "Layout" => "manual/layout.md",
                "Tiling MatMul" => "manual/tiling_matmul.md",
                "Data Movement" => [
                    "Global Memory & Shared Memory" => "manual/datamovement/gs.md",
                ]
            ],
            "API Reference" => [
                    "Layout" => "api/layout.md",
                    "MoYeArray" => "api/array.md",
                    "Tiling" => "api/tiling.md",
                ],
         ])

deploydocs(; repo="github.com/YichengDWu/MoYe.jl", push_preview=true, devbranch="main")
