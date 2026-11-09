const repository = normpath(joinpath(@__DIR__, ".."))
const baseline_directory = joinpath(repository, "results", "experiments", "sparse_adaptation")
const dataset_file = joinpath(repository, "Data", "thesis_v4", "uniform.jld2")
const pretrained_file = joinpath(repository, "Trained_models", "flux_correction_pretrained.bson")
const output_directory = joinpath(repository, "results", "experiments", "adaptation_profiling")
const fold = 1
const samples = 10
const warmup = 3
const methods = [:full, :first_two, :last_two, :output, :lora, :film]
using Flux, LinearAlgebra, Dates

include(joinpath(@__DIR__, "experiment_dependencies.jl"))
include(joinpath(@__DIR__, "src", "sparse_campaign.jl"))
include(joinpath(@__DIR__, "src", "adaptation_profiling.jl"))

function main()
    # Measure saved adaptation methods with a fixed BLAS thread count.
    BLAS.set_num_threads(2)
    output = output_directory
    baseline = baseline_directory
    results = Dict{Symbol, Any}()
    models = Dict{Symbol, Any}()
    # Load one saved fit per method without training again.
    for method in methods
        directory = SparseCampaign.run_directory(baseline, method, :balanced, :uniform, :both,
            0.0, (fold=fold, repeat=1))
        results[method] = JLD2.load(joinpath(directory, "result.jld2"), "result")
        models[method] = Surge.load_model(joinpath(directory, "best_model.bson"))
    end
    # Use the same data split and reference scales for all timing comparisons.
    configuration = results[:full].configuration
    split = results[:full].split
    dataset = Surge.load_surge_dataset(basename(dataset_file);
        data_path=dirname(dataset_file))
    training_data, validation_data = Surge.prepare_training_split(dataset, split, configuration)
    pretrained_model = Surge.load_model(pretrained_file)
    normalizers = Surge.compute_pretrained_error_scales(pretrained_model, training_data,
        configuration.training, configuration.stations)
    # Use the same first batch and pretrained starting states for all methods.
    windows = Surge.enumerate_training_windows(training_data, configuration.training)
    raw_batch = first(Surge.prepare_window_batches(training_data, windows,
        configuration.training, configuration.stations))
    rollout_states = Surge.precompute_rollout_states(pretrained_model, training_data)
    batch = Surge.prepare_objective_batch!(raw_batch, rollout_states, configuration.training,
        configuration.training.rollout_horizons, normalizers)
    objective = model -> Surge.rollout_objective(model, batch.state, batch.forcing,
        batch.targets, configuration.stations, batch.horizons, batch.normalizers)
    # Warm up the separate operations and time them on the same batch.
    actions = AdaptationProfiling.build_actions(models, methods, objective,
        training_data, validation_data, configuration.stations)
    rows = AdaptationProfiling.measure_actions(actions, methods; fold=fold, samples=samples, warmup=warmup)

    # Save timing samples, summaries and environment details.
    mkpath(output)
    Surge.save_table(joinpath(output, "samples.tsv"), rows)
    summary = AdaptationProfiling.summarize_measurements(rows, methods, propertynames(actions[:full]))
    Surge.save_table(joinpath(output, "summary.tsv"), summary)

    open(joinpath(output, "environment.txt"), "w") do io
        println(io,"Date: $(now())\nJulia: $VERSION\nJulia threads: $(Threads.nthreads())\nBLAS threads: $(BLAS.get_num_threads())\nCPU: $(Sys.CPU_NAME)\nFold: $fold\nRepeat: 1\nWarmups: $warmup\nSamples: $samples\nBaseline: $baseline\nBatch windows: $(raw_batch.windows)\nHorizons: $(batch.horizons)")
    end
    println("Saved timing and allocation tables to $output. No training fits executed.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
