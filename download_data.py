import quilt3

pkg = quilt3.Package.browse(
    "aics/pipeline_integrated_single_cell",
    registry="s3://allencell",
    top_hash="7fd488f05ec41968607c7263cb13b3e70812972a24e832ef6f72195bdd35f1b2",
)

pkg.fetch("./data/")
