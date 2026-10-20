# CPU profiling in Section 6.2 and Appendix A.2. No training fits are run.

# 1. Paths and the measurement settings recorded in the thesis ---------------
# Resolve dataset and model paths from the repository containing this script.
const repository = normpath(joinpath(@__DIR__, ".."))
# Read the six already fitted multi-step adaptation methods from the main campaign.
const baseline_directory = joinpath(repository, "results", "thesis_reproduction", "main")
# The same uniform-drag ensemble supplies a shared profiling batch and validation events.
# Sparse supervision uses water level h [m]; velocity u [m/s] is evolved but not supervised.
const dataset_file = joinpath(repository, "Data", "thesis_v4", "uniform.jld2")
# Pretrained weights define the fixed initial-state cache and loss-normalisation scales.
const pretrained_file = joinpath(repository, "Trained_models", "flux_correction_pretrained.bson")
# Save timing/allocation tables and CPU call-stack reports without modifying fitted models.
const output_directory = joinpath(repository, "results", "thesis_reproduction", "adaptation_profile")
# Use the reported first outer fold and, below, its first training repeat.
const fold = 1
# Collect ten measurements after three warm-up calls for each method and measured phase.
const samples = 10
const warmup = 3
# Separately sample gradient-computation call stacks for about ten seconds per method.
const profile_seconds = 10.0
# Include the same full, layer-restricted, low-rank and FiLM models as Experiment 1.
const methods = [:full, :first_two, :last_two, :output, :lora, :film]
# Profile samples CPU call stacks; LinearAlgebra provides control over BLAS thread counts.
using Profile, LinearAlgebra

# Only load numerical/training packages when profiling is actually requested.
if "--run" in ARGS
    include(joinpath(@__DIR__, "experiment_io.jl"))
end

function main()
    # 2. Load saved models and prepare the same four-window batch for all -----
    # Match the reported BLAS thread count; the command line controls Julia's own thread count.
    BLAS.set_num_threads(2)
    # Use short local names for the configured output, baseline and stack-sampling duration.
    output = output_directory
    baseline = baseline_directory
    seconds = profile_seconds
    # Store each method's run directory, metadata and selected model under its method Symbol.
    directories = Dict{Symbol, String}()
    results = Dict{Symbol, Any}()
    models = Dict{Symbol, Any}()
    # Load one saved fit per method; no additional adaptation training is performed.
    for method in methods
        # Scientific directory keys identify clean uniform-drag, four-gauge, multi-step fits.
        name = "$(method)__balanced__uniform__both__noise0cm"
        directory = joinpath(baseline, "runs", name, "fold_$fold", "repeat_1")
        # Metadata supplies the original configuration/split; BSON supplies the selected weights.
        directories[method] = directory
        results[method] = JLD2.load(joinpath(directory, "result.jld2"), "result")
        models[method] = Surge.load_model(joinpath(directory, "best_model.bson"))
    end
    # Use the full-fit settings and split to prepare a batch shared by every method.
    configuration = results[:full].configuration
    split = results[:full].split
    # Return complete states/grid/parameters plus the native snapshot interval in seconds.
    dataset, time_step = ConservativePilot.load_dataset(dataset_file)
    # Select this split's events and prepare its original sparse h observations and training duration.
    training_data, validation_data = Surge.prepare_training_split(dataset, split, configuration)
    # Restore the common source model used to calculate fixed loss scales and batch starting states.
    pretrained_model = Surge.load_model(pretrained_file)
    # Compute pretrained MSE at each supervised lead using training observations only.
    normalizers = Surge.compute_pretrained_error_scales(pretrained_model, training_data,
        configuration.training, configuration.stations)
    # List the same non-overlapping (event, window-start) pairs used during training.
    windows = Surge.enumerate_training_windows(training_data, configuration.training)
    # Build forcing/target arrays and select the first batch of at most four windows.
    # Prescribed forcing features are depth/30, wind stress and boundary flags; targets are sparse h.
    raw_batch = first(Surge.prepare_window_batches(training_data, windows,
        configuration.training, configuration.stations))
    # Forecast training events once to obtain model-generated starting states at every step.
    rollout_states = Surge.precompute_rollout_states(pretrained_model, training_data)
    # Fill the batch's starting states and attach leads 16/32/64/128 and their fixed loss scales.
    batch = Surge.prepare_objective_batch!(raw_batch, rollout_states, configuration.training,
        configuration.training.rollout_horizons, normalizers)
    # Define one scalar multi-step observation loss with fixed inputs/targets for all timed models.
    # Calling objective(model) advances its own predictions through the shared batch.
    objective = model -> Surge.rollout_objective(model, batch.state, batch.forcing,
        batch.targets, configuration.stations, batch.horizons, batch.normalizers)
    # Every method sees exactly the same batch, initial states, forcing and targets.
    # Store small callable operations so each phase can be warmed and timed independently.
    actions = Dict{Symbol, Any}()
    for method in methods
        # Measure this saved validation-selected model, not a newly initialised version of it.
        model = models[method]
        # Calculate a gradient once for the optimiser-only benchmark; [1] selects the model argument.
        gradient = Flux.gradient(objective, model)[1]
        # Updates affect only a disposable copy, never the benchmark model/checkpoint.
        update_model = deepcopy(model)
        # Allocate Adam's moment state for the disposable model before timing optimiser steps.
        optimizer = Flux.setup(Flux.Adam(3e-6), update_model)
        actions[method] = (
            # Materialise effective convolution kernels, including any low-rank updates.
            kernel_materialization=() -> Surge.effective_model(model),
            # Measure a forward rollout/loss without differentiating it.
            forward=() -> objective(model),
            # Measure the forward rollout together with reverse-mode gradient computation.
            forward_reverse=() -> Flux.gradient(objective, model),
            # Apply the precomputed gradient to the disposable model; do not recompute it here.
            optimizer=() -> Flux.update!(optimizer, update_model, gradient),
            # Measure regeneration of all model-generated training-window starting states.
            state_regeneration=() -> Surge.precompute_rollout_states(model, training_data),
            # Measure continuous sparse-observation validation on this split's validation events.
            validation=() -> Surge.validation_loss(model, validation_data, configuration.stations))
    end
    # 3. Warm each measured phase before collecting timing samples ------------
    # Named-tuple field names give the common list of operations to measure for every method.
    phases = propertynames(actions[:full])
    println("Warming all six methods; compilation is excluded from measured samples.")
    # Visit every method/phase and call it three times before retaining measurements.
    for method in methods
        for phase in phases
            for warmup_index in 1:warmup
                # Select the zero-argument function for this phase and execute it to warm compilation.
                getproperty(actions[method], phase)()
            end
        end
    end
    # 4. Save samples and medians in the original TSV schemas ----------------
    # Create the output directory only after all measured operations have been prepared and warmed.
    mkpath(output)
    # Each row will represent one sample of one method/phase combination.
    rows = NamedTuple[]
    # Fix the random order of measurements to make the profiling protocol reproducible.
    random_generator = MersenneTwister(20261016)
    # Repeat the complete method/phase comparison ten times.
    for sample in 1:samples
        # Shuffle methods, then phase order within each method, to vary measurement order.
        for method in shuffle(random_generator, methods)
            for phase in shuffle(random_generator, collect(phases))
                # Collect old allocations before timing; this collection is outside the measured call.
                GC.gc()
                # @timed reports elapsed seconds, allocated bytes and garbage-collection time.
                measurement = @timed getproperty(actions[method], phase)()
                # Retain raw measurements so medians and spread can be inspected or recalculated.
                push!(rows, (method=string(method), fold=fold, sample=sample, phase=string(phase),
                    seconds=measurement.time, allocated_bytes=measurement.bytes, gc_seconds=measurement.gctime))
            end
        end
        # Show progress after all six methods and six phases in this sample are complete.
        println("Completed timing sample $sample/$samples")
    end
    # Save every raw timing/allocation measurement in the original TSV format.
    Surge.save_table(joinpath(output, "samples.tsv"), rows)
    # Reduce repeated measurements to one summary row per method and phase.
    summary = NamedTuple[]
    for method in methods
        for phase in phases
            # Select only the ten samples of this particular operation and model.
            selected = filter(row -> row.method == string(method) && row.phase == string(phase), rows)
            # Extract elapsed times in seconds; allocated-byte medians are calculated separately.
            times = [row.seconds for row in selected]
            # Report the median and 25th/75th percentiles, describing typical cost and sample spread.
            push!(summary, (method=string(method), phase=string(phase), median_seconds=median(times),
                q25_seconds=quantile(times, 0.25), q75_seconds=quantile(times, 0.75),
                median_allocated_bytes=median([row.allocated_bytes for row in selected])))
        end
    end
    # Save the same summary schema used for the thesis's phase-cost comparisons.
    Surge.save_table(joinpath(output, "summary.tsv"), summary)
    # 5. Save the original CPU sampling reports and environment record --------
    # CPU sampling identifies call stacks; unlike wall time it does not measure allocations.
    # Profile each method separately so its CPU sample stacks do not mix with other methods.
    for method in methods
        if seconds == 0
            # A zero duration skips optional stack sampling while retaining the timing tables.
            break
        end
        # Discard previous CPU samples and calculate when this method's sampling window ends.
        Profile.clear()
        deadline = time() + seconds
        # Repeatedly compute the shared-batch gradient while Julia samples the active call stack.
        Profile.@profile while time() < deadline
            actions[method].forward_reverse()
        end
        # Export a tree showing nested callers; the do block closes the text file after writing.
        open(joinpath(output, "$(method)_stacks.txt"), "w") do io
            Profile.print(io; format=:tree, C=true)
        end
        # Also export a flat list ranked by sampled call counts, including C-library frames.
        open(joinpath(output, "$(method)_flat.txt"), "w") do io
            Profile.print(io; format=:flat, C=true, sortedby=:count)
        end
    end
    # Record hardware/runtime/thread counts and the exact batch/lead settings needed to interpret costs.
    open(joinpath(output, "environment.txt"), "w") do io
        println(io,"Date: $(now())\nJulia: $VERSION\nJulia threads: $(Threads.nthreads())\nBLAS threads: $(BLAS.get_num_threads())\nCPU: $(Sys.CPU_NAME)\nFold: $fold\nRepeat: 1\nWarmups: $warmup\nSamples: $samples\nProfile seconds/method: $seconds\nBaseline: $baseline\nBatch windows: $(raw_batch.windows)\nHorizons: $(batch.horizons)")
    end
    # Display where measurements were written; all selected model files remain untouched.
    println("Saved timing tables and CPU profiles to $output. No training fits executed.")
end

# Start profiling only with --run; otherwise print the measurement plan and thread requirement.
if "--run" in ARGS
    main()
else
    println("CPU profiling: six saved methods, fold 1/repeat 1, three warmups, ten samples.")
    println("Run with --threads=1; BLAS uses two threads. Add --run to execute.")
    println("Output: $output_directory")
end
