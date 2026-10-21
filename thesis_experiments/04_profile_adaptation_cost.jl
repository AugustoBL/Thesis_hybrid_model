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

function main()
    BLAS.set_num_threads(2)
    output = output_directory
    baseline = baseline_directory
    results = Dict{Symbol, Any}()
    models = Dict{Symbol, Any}()
    # Load one saved fit per method; no additional adaptation training is performed.
    for method in methods
        name = "$(method)__balanced__uniform__both__noise0cm"
        directory = joinpath(baseline, "runs", name, "fold_$fold", "repeat_1")
        results[method] = JLD2.load(joinpath(directory, "result.jld2"), "result")
        models[method] = Surge.load_model(joinpath(directory, "best_model.bson"))
    end
    configuration = results[:full].configuration
    split = results[:full].split
    dataset = Surge.load_burgers_ensemble(basename(dataset_file);
        data_path=dirname(dataset_file), relative_times=true)
    training_data, validation_data = Surge.prepare_training_split(dataset, split, configuration)
    pretrained_model = Surge.load_model(pretrained_file)
    normalizers = Surge.compute_pretrained_error_scales(pretrained_model, training_data,
        configuration.training, configuration.stations)
    windows = Surge.enumerate_training_windows(training_data, configuration.training)
    raw_batch = first(Surge.prepare_window_batches(training_data, windows,
        configuration.training, configuration.stations))
    rollout_states = Surge.precompute_rollout_states(pretrained_model, training_data)
    batch = Surge.prepare_objective_batch!(raw_batch, rollout_states, configuration.training,
        configuration.training.rollout_horizons, normalizers)
    objective = model -> Surge.rollout_objective(model, batch.state, batch.forcing,
        batch.targets, configuration.stations, batch.horizons, batch.normalizers)
    # Isolated microbenchmarks retain allocations/GC, standalone forward passes,
    # and optimizer-only costs, which train_model! phase totals do not measure.
    actions = Dict{Symbol, Any}()
    for method in methods
        model = models[method]
        gradient = Flux.gradient(objective, model)[1]
        update_model = deepcopy(model)
        optimizer = Flux.setup(Flux.Adam(3e-6), update_model)
        actions[method] = (
            kernel_materialization=() -> Surge.effective_model(model),
            forward=() -> objective(model),
            forward_reverse=() -> Flux.gradient(objective, model),
            optimizer=() -> Flux.update!(optimizer, update_model, gradient),
            state_regeneration=() -> Surge.precompute_rollout_states(model, training_data),
            validation=() -> Surge.validation_loss(model, validation_data, configuration.stations))
    end

    phases = propertynames(actions[:full])
    println("Warming all six methods; compilation is excluded from measured samples.")
    for method in methods
        for phase in phases
            for warmup_index in 1:warmup
                getproperty(actions[method], phase)()
            end
        end
    end

    mkpath(output)
    rows = NamedTuple[]
    random_generator = MersenneTwister(20261016)
    for sample in 1:samples
        for method in shuffle(random_generator, methods)
            for phase in shuffle(random_generator, collect(phases))
                GC.gc()
                measurement = @timed getproperty(actions[method], phase)()
                push!(rows, (method=string(method), fold=fold, sample=sample, phase=string(phase),
                    seconds=measurement.time, allocated_bytes=measurement.bytes, gc_seconds=measurement.gctime))
            end
        end
        println("Completed timing sample $sample/$samples")
    end
    Surge.save_table(joinpath(output, "samples.tsv"), rows)
    summary = NamedTuple[]
    for method in methods
        for phase in phases
            selected = filter(row -> row.method == string(method) && row.phase == string(phase), rows)
            times = [row.seconds for row in selected]
            push!(summary, (method=string(method), phase=string(phase), median_seconds=median(times),
                q25_seconds=quantile(times, 0.25), q75_seconds=quantile(times, 0.75),
                median_allocated_bytes=median([row.allocated_bytes for row in selected])))
        end
    end
    Surge.save_table(joinpath(output, "summary.tsv"), summary)

    open(joinpath(output, "environment.txt"), "w") do io
        println(io,"Date: $(now())\nJulia: $VERSION\nJulia threads: $(Threads.nthreads())\nBLAS threads: $(BLAS.get_num_threads())\nCPU: $(Sys.CPU_NAME)\nFold: $fold\nRepeat: 1\nWarmups: $warmup\nSamples: $samples\nBaseline: $baseline\nBatch windows: $(raw_batch.windows)\nHorizons: $(batch.horizons)")
    end
    println("Saved timing and allocation tables to $output. No training fits executed.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
