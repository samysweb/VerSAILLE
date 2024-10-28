using SNNT
using JLD
using Glob
using Images
using ImageTransformations
using Plots

function summarize_and_load(folder, prefix)
    println("Loading results from $folder/$prefix-*.jld")
    results = []
    metadata = nothing
    for file in glob("$prefix-*.jld",folder)
        if occursin("summary",file)
            continue
        end
        cur_results = load(file)
        if haskey(cur_results,"backup_meta")
            metadata = cur_results["backup_meta"]
        end
        append!(results,cur_results["result"])
    end
    result_summary = SNNT.VerifierInterface.reduce_results(results)
    save("$folder/$prefix-summary.jld","result",result_summary,"args",metadata)
    return (result_summary, metadata)
end