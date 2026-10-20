# Experiments 1-4: parameterisation, forecast leads, geometry/drag, and noise.
# Run with --run. Without it, print the plan without loading data or packages.

# 1. Paths and common experimental settings ----------------------------------
# Resolve every path from the repository containing this script.
const repository = normpath(joinpath(@__DIR__, ".."))
# Each file contains the matched trajectories for one drag profile or the source control.
const data_directory = joinpath(repository, "Data", "thesis_v4")
# All final fits start from this same source-trained flux-correction CNN.
const pretrained_file = joinpath(repository, "Trained_models", "flux_correction_pretrained.bson")
# New fits and analysis tables are kept apart from the historical thesis results.
const output_directory = joinpath(repository, "results", "thesis_reproduction", "main")

# Compare unrestricted, layer-restricted, low-rank and channel-wise adaptation.
# first_two/last_two/output update only the named convolutional layers.
# LoRA updates low-rank weight factors and biases; FiLM updates feature scales and shifts.
const adaptation_methods = [:full, :output, :last_two, :first_two, :lora, :film]
# These five loss/gradient configurations compete for the fold's temporal choice.
const selection_objectives = [:short, :long, :balanced, :curriculum, :detached]
# These extra comparisons are evaluated but cannot change that temporal choice.
const additional_objectives = [:fixed32, :fixed64, :one_step]
# Layout names map to observed h-cell indices on the unchanged 100-cell domain.
# h is water level [m]; u is velocity [m/s] and remains unobserved during adaptation.
const station_layouts = Dict(
    # Four gauges at both ends, four at one end, eight at both ends, or four spread out.
    :both => [2, 4, 97, 99],
    :one => [2, 4, 6, 8],
    :eight => [2, 4, 6, 8, 93, 95, 97, 99],
    :spread => [10, 37, 64, 91],
)
# Test only the drag-profile/layout pairs reported in Experiment 3.
const geometry_cases = [
    (:uniform, :both), (:uniform, :one), (:uniform, :eight), (:uniform, :spread),
    (:interior, :both), (:interior, :one), (:interior, :spread), (:boundary, :both),
]
# Standard deviations of independent Gaussian errors added to sparse h observations.
const noise_standard_deviations = [0.01, 0.05, 0.10, 0.20] # metres
# Adam's fixed learning rate, first/second moment decays, and denominator stabiliser.
const optimizer_settings = (learning_rate=3e-6, betas=(0.9, 0.999), epsilon=1e-8)
const training_settings = (
    # An epoch visits all training windows; a batch contains up to four event-window pairs.
    epochs=200,
    batch_size=4,
    # Leads are measured in native model transitions, not in observation samples.
    rollout_horizons=[16, 32, 64, 128],
    # Keep the same non-overlapping window starts even for shorter supervised leads.
    window_stride=128,
    # balanced averages losses over the active leads; endpoint uses only one lead.
    loss_mode=:balanced,
    endpoint_horizon=128,
    # Divide each lead's MSE by the pretrained model's fixed training-error scale.
    normalize_loss=true,
    # Curriculum variants introduce longer leads at these fractions of the epoch budget.
    curriculum_enabled=false,
    curriculum_boundaries=[0.25, 0.375, 0.5], # 50/25/25/100 epochs
    # Full gradients are the default; detached variants retain gradient_steps final transitions.
    detach_gradients=false,
    gradient_steps=16,
    # Keep the legacy field in saved configurations; current detachment uses the settings above.
    native_one_step=false,
    # Validate at epoch zero, every five epochs and the final epoch; disable early stopping.
    validation_interval=5,
    patience=0,
    # The trainer also refreshes window-start states each epoch; this is its within-epoch interval.
    cache_refresh_interval=25,

)

# A plan-only invocation does not load models, datasets or the training packages.
if "--run" in ARGS
    # Load shared training/diagnostic functions and the original timed result writer.
    include(joinpath(@__DIR__, "experiment_io.jl"))
    # Reuse the original nested LoRA rank-selection procedure rather than implementing it here.
    include(joinpath(repository, "lora_rank_optimisation", "rank_selection.jl"))
end

# 2. Translate the original objective names into explicit trainer settings ---
function objective_settings(objective)
    # Return the common multi-step configuration without changing any fields.
    if objective == :balanced
        return training_settings
    elseif objective == :curriculum
        # merge changes only this field, keeping all other shared settings intact.
        return merge(training_settings, (curriculum_enabled=true,))
    end

    # Endpoint comparisons use one supervised lead: 16, 32, 64 or the default 128.
    endpoint = 128
    if objective == :short
        endpoint = 16
    elseif objective == :fixed32
        endpoint = 32
    elseif objective == :fixed64
        endpoint = 64
    end
    # Keep the historical one_step result name for analysis compatibility.
    # Current src expresses it directly as 127 detached steps and one
    # differentiable transition, with loss only at the step-128 endpoint.
    # Both truncated-gradient comparisons still forecast all the way to lead 128.
    detach_gradients = objective == :detached || objective == :one_step
    # Retaining 16 gradients detaches the first 112 transitions of that rollout.
    gradient_steps = 16
    if objective == :one_step
        # Retaining one gradient detaches the first 127 transitions instead.
        gradient_steps = 1
    end
    # Override only the objective-specific fields in the common training record.
    return merge(training_settings, (
        loss_mode=:endpoint,
        endpoint_horizon=endpoint,
        detach_gradients=detach_gradients,
        gradient_steps=gradient_steps,
    ))
end

# Assemble the complete settings for one condition. ranks holds a rank vector per outer fold.
function configuration(method, objective, profile, layout, noise_std, ranks)
    return (
        # Retain the original campaign identity and JLD2 configuration fields.
        experiment=0, name="thesis_flux_correction",
        output_directory=output_directory,
        # The profile chooses the physical trajectories; layout chooses which h cells are observed.
        dataset_file=joinpath(data_directory, "$profile.jld2"),
        pretrained_file=pretrained_file,
        # Observe every 16 native steps within the first 1,024 transitions of each fitting event.
        measurement_gap=16, training_steps=1024,
        stations=station_layouts[layout], observation_noise_std=noise_std,
        # Distinct offsets give fixed but independent training and validation noise draws.
        training_noise_seed_offset=710000, validation_noise_seed_offset=720000,
        # The split seed plus this offset determines the training batch-order RNG.
        optimizer_seed_offset=10000, optimizer=optimizer_settings,
        # Configure the trainable parameter space and resolve the named rollout objective.
        adaptation=(method=method, ranks=ranks), training=objective_settings(objective),
    )
end

# A scientific key reuses the same fit when it belongs to more than one stage.
# Stage-specific index rows remain separate, as in the original programme.
function run_condition!(jobs, pretrained_model, datasets, ranks,
        stage, method, objective, profile, layout, noise_std, split)
    # Convert metres to integer centimetres only for the human-readable directory name.
    noise_cm = round(Int, noise_std * 100)
    # Excluding stage from this name lets identical conditions reuse the same fitted model.
    name = "$(method)__$(objective)__$(profile)__$(layout)__noise$(noise_cm)cm"
    # A fold/repeat subdirectory identifies one independent fit of this condition.
    directory = joinpath(output_directory, "runs", name,
        "fold_$(split.fold)", "repeat_$(split.repeat)")
    # Collect the model, data, observation and training settings for this fit.
    settings = configuration(method, objective, profile, layout, noise_std, ranks)
    println("Stage $stage: $name, fold $(split.fold), repeat $(split.repeat)")
    # Train, resume or load this fit; return its validation-selected epoch/loss and timings.
    result = fit_timed_run(pretrained_model, datasets[profile], split, settings, directory)
    # Append one stage-specific index row; repeated conditions can point to the same directory.
    push!(jobs, (stage=stage, method=method, temporal=objective, profile=profile,
        layout=layout, noise_std=noise_std, fold=split.fold, repeat=split.repeat,
        directory=directory, best_validation_loss=result.best_validation_loss,
        best_epoch=result.best_epoch, completed_epochs=result.completed_epochs,
        training_seconds=result.training_seconds, preparation_seconds=result.preparation_seconds))
    # Update the readable campaign index after every condition finishes.
    Surge.save_table(joinpath(output_directory, "run_index.tsv"), jobs)
    # Selection below compares this sparse validation MSE, never the test forecast error.
    return result.best_validation_loss
end

# 3. Load the original ensembles and select LoRA ranks within training pools --
function main()
    # Restore the source-trained model weights from the original BSON checkpoint.
    pretrained_model = Surge.load_model(pretrained_file)
    # Index complete event ensembles by drag profile; control has no additional drag.
    datasets = Dict{Symbol, Any}()
    # Load the three target systems and their paired source-system reference once.
    for profile in (:uniform, :interior, :boundary, :control)
        # The loader returns states/grid/parameters and the stored-state interval in seconds.
        dataset, time_step = ConservativePilot.load_dataset(joinpath(data_directory, "$profile.jld2"))
        datasets[profile] = dataset
    end
    # Generate matched train/validation/test memberships and seeds for all methods.
    # Each of five outer folds has three draws of two training events and four validation events.
    event_splits = Surge.make_event_splits(18;
        fold_count=5, folds=collect(1:5), repeats=collect(1:3),
        training_event_count=2, validation_event_count=4,
        selection_seed=42000, training_seed=420000)

    # Fingerprint this reproduction's code and inputs, keeping historical
    # manifests and results untouched. The manifest retains its original keys.
    source_files = [@__FILE__, joinpath(@__DIR__, "experiment_io.jl"),
        joinpath(repository, "thesis_experiments", "shared", "conservative_pilot_utils.jl"),
        joinpath(repository, "lora_rank_optimisation", "rank_selection.jl"),
        joinpath(repository, "Starting_networks", "Project.toml"),
        joinpath(repository, "Starting_networks", "Manifest.toml")]
    # Include every Julia source file, in sorted order, in the provenance record.
    append!(source_files, sort(filter(path -> endswith(path, ".jl"),
        readdir(joinpath(repository, "src"); join=true))))
    # SHA-256 digests record exact code, checkpoint and dataset bytes; they do not alter inputs.
    signature = (
        source_hashes=[bytes2hex(sha256(read(path))) for path in source_files],
        checkpoint=bytes2hex(sha256(read(pretrained_file))),
        data=[bytes2hex(sha256(read(joinpath(data_directory, "$profile.jld2"))))
            for profile in (:uniform, :interior, :boundary, :control)],
    )
    # Create the campaign directory, then write its manifest only on the first invocation.
    mkpath(output_directory)
    manifest_path = joinpath(output_directory, "manifest.jld2")
    if !isfile(manifest_path)
        # Save common hyperparameters, input paths, Julia version and creation time alongside hashes.
        JLD2.jldsave(manifest_path; signature=signature, training=training_settings,
            optimizer=optimizer_settings, checkpoint=pretrained_file,
            data=data_directory, julia_version=string(VERSION), created=string(now()))
    end

    # Each stored TOML entry maps an outer-fold number to five selected convolutional ranks.
    rank_file = joinpath(output_directory, "rank_selection", "selected_ranks.toml")
    ranks = Dict{Int, Vector{Int}}()
    # A rerun reuses completed rank selection rather than fitting its candidates again.
    if isfile(rank_file)
        for (fold, selected_ranks) in TOML.parsefile(rank_file)["folds"]
            # Convert TOML's string fold keys and numeric arrays into the types expected by Surge.
            ranks[parse(Int, fold)] = Int.(selected_ranks)
        end
    else
        # Rank selection uses clean uniform-drag data and writes to a separate subdirectory.
        rank_configuration = merge(configuration(:full, :balanced, :uniform, :both, 0.0, nothing),
            (output_directory=joinpath(output_directory, "rank_selection"),))
        # Supply one outer-pool definition per fold; the selector creates its own inner repeats.
        pilot_splits = filter(split -> split.repeat == 1, event_splits)
        # A short full-fit pilot supplies singular-value spectra of layer weight updates.
        # Candidate rank vectors retain 80/90/95/99% squared singular-value energy.
        # Inner validation selects the smallest model within one standard error of the best mean.
        ranks = LoRARankOptimisation.select_ranks(pretrained_model, datasets[:uniform], pilot_splits;
            settings=rank_configuration,
            # Ten pilot epochs generate candidates; candidate fits stop at 50 epochs or patience ten.
            pilot_training=merge(training_settings, (epochs=10,)),
            candidate_training=merge(training_settings,
                (epochs=50, validation_interval=1, patience=10)),
            thresholds=[0.8, 0.9, 0.95, 0.99],
            # Inner training/validation events come only from the outer-training pool.
            # Separate seeds fix the held-out inner events, training draws and optimisation.
            inner_split_settings=(repeats=collect(1:3), training_event_count=2,
                validation_event_count=2, holdout_seed=62000,
                selection_seed=63000, training_seed=640000))
    end

    # 4. Fit Experiments 1-4, selecting only on outer validation ---------------
    # Accumulate the index records needed for evaluation and the stage-specific tables.
    jobs = NamedTuple[]
    for fold in 1:5
        # All candidate conditions in this fold share the same three repeat splits.
        fold_splits = filter(split -> split.fold == fold, event_splits)

        # Experiment 1: common multi-step training for six parameterisations.
        # Store one mean validation loss per adaptation method.
        method_scores = NamedTuple[]
        for method in adaptation_methods
            # Collect the three repeated-fit losses for this candidate method.
            validation_losses = Float64[]
            for split in fold_splits
                # Fit this repeat on clean uniform drag with the default four gauges.
                loss = run_condition!(jobs, pretrained_model, datasets, ranks,
                    1, method, :balanced, :uniform, :both, 0.0, split)
                push!(validation_losses, loss)
            end
            # Average validation MSE across repeats before comparing methods.
            push!(method_scores, (candidate=method, validation=mean(validation_losses)))
        end
        # Lowest loss wins; the method name provides a deterministic tie-break.
        sort!(method_scores; by=score -> (score.validation, string(score.candidate)))
        # Save every candidate's score, then carry the best method into Experiment 2.
        Surge.save_table(joinpath(output_directory, "selections", "method_fold_$fold.tsv"), method_scores)
        selected_method = first(method_scores).candidate

        # Experiment 2: five candidates participate in selection. Fixed-32,
        # fixed-64 and the 127-detached-step comparison are additional controls.
        # Store repeat-averaged validation losses for the five temporal candidates.
        objective_scores = NamedTuple[]
        for objective in selection_objectives
            # Compare temporal objectives using the same three training-event draws.
            validation_losses = Float64[]
            for split in fold_splits
                loss = run_condition!(jobs, pretrained_model, datasets, ranks,
                    2, selected_method, objective, :uniform, :both, 0.0, split)
                push!(validation_losses, loss)
            end
            # A single unusually good repeat cannot replace the mean over all three fits.
            push!(objective_scores, (candidate=objective, validation=mean(validation_losses)))
        end
        # Save the sorted validation comparison and carry its winner into Experiments 3 and 4.
        sort!(objective_scores; by=score -> (score.validation, string(score.candidate)))
        Surge.save_table(joinpath(output_directory, "selections", "temporal_fold_$fold.tsv"), objective_scores)
        selected_objective = first(objective_scores).candidate
        # Fit each additional temporal control for every repeat without adding it to selection.
        for objective in additional_objectives
            for split in fold_splits
                run_condition!(jobs, pretrained_model, datasets, ranks,
                    2, selected_method, objective, :uniform, :both, 0.0, split)
            end
        end

        # Experiment 3: keep the selected method/objective when geometry or
        # drag changes. Each condition still selects its own best checkpoint.
        # Repeat each reported physical-shift/observation-layout combination on the paired splits.
        for (profile, layout) in geometry_cases
            for split in fold_splits
                run_condition!(jobs, pretrained_model, datasets, ranks,
                    3, selected_method, selected_objective, profile, layout, 0.0, split)
            end
        end

        # Experiment 4: fresh adaptation from pretrained weights for each
        # noise level; training/validation noise have independent fixed seeds.
        # Change only observation noise, keeping the selected method, objective and event splits.
        for noise_std in noise_standard_deviations
            for split in fold_splits
                run_condition!(jobs, pretrained_model, datasets, ranks,
                    4, selected_method, selected_objective, :uniform, :both, noise_std, split)
            end
        end
    end

    # 5. Evaluate held-out events and matched source retention ----------------
    # The index can mention one shared fit in several stages; forecast it only once.
    evaluated_directories = Set{String}()
    for job in jobs
        if job.directory in evaluated_directories
            # Its forecasts already exist from an earlier stage using this same fit.
            continue
        end
        # Mark this directory before preparing its target and source-retention reports.
        push!(evaluated_directories, job.directory)
        report_path = joinpath(job.directory, "diagnostics", "report_metrics.jld2")
        # Reuse saved diagnostics; otherwise forecast the held-out events from their initial states.
        if !isfile(report_path)
            # Compare selected and pretrained models against clean full states through 72 hours.
            # Cells 9:92 define the interior diagnostic; boundaries divide early/late error intervals.
            ExperimentDiagnostics.diagnose_saved_run(job.directory, datasets[job.profile];
                pretrained_model=pretrained_model, forecast_hours=72.0,
                interior_indices=collect(9:92), interval_boundaries=[0.0, 12.0, 24.0, 48.0, 72.0])
        end
        # Also measure how target adaptation affects matched no-extra-drag source events.
        retention_directory = joinpath(job.directory, "source_retention")
        if !isfile(joinpath(retention_directory, "diagnostics", "report_metrics.jld2"))
            # Copy metadata and selected weights so the existing diagnostic reader can load them.
            mkpath(retention_directory)
            cp(joinpath(job.directory, "result.jld2"), joinpath(retention_directory, "result.jld2"); force=true)
            cp(joinpath(job.directory, "best_model.bson"), joinpath(retention_directory, "best_model.bson"); force=true)
            # Keep this original source-retention evaluation at 12 hours; do not retrain the model.
            ExperimentDiagnostics.diagnose_saved_run(retention_directory, datasets[:control];
                pretrained_model=pretrained_model, forecast_hours=12.0,
                interior_indices=collect(9:92), interval_boundaries=[0.0, 12.0])
        end
    end

    # 6. Save the original analysis tables -----------------------------------
    # Collect interval errors for every event, repeat, model and stage-specific condition.
    metrics = NamedTuple[]
    for job in jobs
        # Load the original result record containing per-event series and interval summaries.
        report = JLD2.load(joinpath(job.directory, "diagnostics", "report_metrics.jld2"), "result")
        # Use the varied factor as each stage's table label: method, objective, geometry or noise.
        if job.stage == 1
            condition = string(job.method)
        elseif job.stage == 2
            condition = string(job.temporal)
        elseif job.stage == 3
            condition = "$(job.profile)__$(job.layout)"
        else
            condition = "noise$(round(Int, 100 * job.noise_std))cm"
        end
        # Prefix each diagnostic row with its experiment identity and measured training cost.
        for row in report.summaries
            push!(metrics, (stage=job.stage, condition=condition, fold=job.fold,
                repeat=job.repeat, training_seconds=job.training_seconds, row...))
        end
    end
    # Retain individual event/repeat errors so later analyses can make paired comparisons.
    Surge.save_table(joinpath(output_directory, "event_metrics.tsv"), metrics)

    # Average MSE over repeats within each event before giving events equal
    # weight. Unequal test-fold sizes must not become unequal event weights.
    event_groups = Dict{Any, Vector{NamedTuple}}()
    for row in metrics
        # Leave repeat out of the key so the three fits of one held-out event are grouped together.
        key = (row.stage, row.condition, row.model, row.event, row.from_hours, row.to_hours)
        # get! creates an empty row list the first time a group is encountered.
        push!(get!(event_groups, key, NamedTuple[]), row)
    end
    # The second grouping removes event identity after its repeat average has been calculated.
    condition_groups = Dict{Any, Vector{NamedTuple}}()
    for (key, rows) in event_groups
        # Unpack which condition/model/event and forecast interval these repeats describe.
        stage, condition, model, event, lower, upper = key
        # Average squared errors, not forecasts or per-repeat RMSEs.
        # Squaring bias_rmse preserves bias magnitude even when signed biases cancel.
        record = (mse_h=mean(row.mse_h for row in rows),
            mse_u=mean(row.mse_u for row in rows),
            anomaly_mse=mean(row.anomaly_mse for row in rows),
            bias_squared=mean(row.bias_rmse^2 for row in rows))
        # Add one repeat-averaged record per event to its condition/interval group.
        push!(get!(condition_groups, (stage, condition, model, lower, upper), NamedTuple[]), record)
    end
    summaries = NamedTuple[]
    # Sort keys to give the saved summary a stable, readable row order.
    for (key, rows) in sort(collect(condition_groups); by=pair -> string(first(pair)))
        stage, condition, model, lower, upper = key
        # Count a fit's timing once, regardless of how many test events or intervals it contributes.
        costs = unique([(row.fold, row.repeat, row.training_seconds) for row in metrics
            if row.stage == stage && row.condition == condition])
        # Frozen forecasts have zero adaptation cost; adapted costs average distinct fits equally.
        training_seconds = 0.0
        if model != :pretrained
            training_seconds = mean(last(cost) for cost in costs)
        end
        # Average the event MSEs equally. Readers take sqrt(mse_h) to report water-level RMSE.
        # mse_u is velocity MSE; anomaly_mse removes spatial mean level; bias_squared retains it.
        push!(summaries, (stage=stage, condition=condition, model=model,
            from_hours=lower, to_hours=upper, events=length(rows),
            mse_h=mean(row.mse_h for row in rows), mse_u=mean(row.mse_u for row in rows),
            anomaly_mse=mean(row.anomaly_mse for row in rows),
            bias_squared=mean(row.bias_squared for row in rows), training_seconds=training_seconds))
    end
    # Write the same summary columns consumed by the existing thesis figure builders.
    Surge.save_table(joinpath(output_directory, "summary.tsv"), summaries)
end

# Enter the complete workflow only with --run; otherwise print the planned scope and output path.
if "--run" in ARGS
    main()
else
    println("Experiments 1-4: 360 unique fits, five rank pilots and up to 60 candidate fits.")
    println("Five folds, three repeats, 200 epochs; original JLD2/BSON/TSV outputs.")
    println("Output: $output_directory")
    println("Add --run to execute. Edit the parameters at the top to change the setup.")
end
