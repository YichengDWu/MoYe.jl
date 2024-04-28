using MoYe
using Documenter

DocMeta.setdocmeta!(MoYe, :DocTestSetup, :(using MoYe); recursive=true)

makedocs(; modules=[MoYe],
         authors="MilkshakeForReal <yicheng.wu@ucalgary.ca> and contributors",
         repo="https://github.com/YichengDWu/MoYe.jl/blob/{commit}{path}#{line}",
         warnonly = Documenter.except(:autodocs_block, :cross_references, :docs_block,
         :doctest, :eval_block, :example_block, :footnote,
         :linkcheck_remotes, :linkcheck, :meta_block, :parse_error),
         sitename="MoYe.jl",
         format=Documenter.HTML(; prettyurls=get(ENV, "CI", "false") == "true",
                                canonical="https://YichengDWu.github.io/MoYe.jl",
                                edit_link="main", assets=String[]),
         pages=[
            "Home" => "index.md",
            "Manual" => [
                "Layout" => "manual/layout.md",
                "Array" => "manual/array.md",
                "Broadcasting" => "manual/broadcast.md",
                "MatMul" => "manual/matmul.md",
               # "Data Movement" => [
               #     "Global Memory & Shared Memory" => "manual/datamovement/gs.md",
               # ]
                "TiledCopy & TiledMMA" => "manual/tiled_matmul.md",
                "Memcpy Async" => "manual/async.md",
                "Pipeline" => "manual/pipeline.md",
                "Tensor Cores" => "manual/tensor_core.md",
            ],
            "API Reference" => [
                    "Layout" => "api/layout.md",
                    "MoYeArray" => "api/array.md",
                    "Tiling" => "api/tiling.md",
                    "Data Movement" => "api/copy.md",
                    "MMA/Copy Atoms" => "api/atom.md",
                ],
         ])

deploydocs(; repo="github.com/YichengDWu/MoYe.jl", push_preview=true, devbranch="main")
