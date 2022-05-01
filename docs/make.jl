using BlackBoxModels
using Documenter

DocMeta.setdocmeta!(BlackBoxModels, :DocTestSetup, :(using BlackBoxModels); recursive = true)

makedocs(;
  modules = [BlackBoxModels],
  doctest = true,
  linkcheck = false,
  strict = false,
  authors = "Abel Soares Siqueira <abel.s.siqueira@gmail.com> and contributors",
  repo = "https://github.com/JuliaSmoothOptimizers/BlackBoxModels.jl/blob/{commit}{path}#{line}",
  sitename = "BlackBoxModels.jl",
  format = Documenter.HTML(;
    prettyurls = get(ENV, "CI", "false") == "true",
    canonical = "https://JuliaSmoothOptimizers.github.io/BlackBoxModels.jl",
    assets = ["assets/style.css"],
  ),
  pages = ["Home" => "index.md", "Reference" => "reference.md"],
)

deploydocs(;
  repo = "github.com/JuliaSmoothOptimizers/BlackBoxModels.jl",
  push_preview = true,
  devbranch = "main",
)
