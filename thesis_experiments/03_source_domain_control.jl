# No-shift control in Section 6.5: sparse source-domain long-128 adaptation,
# followed by evaluation on the paired source and uniform-drag target systems.

# 1. Inputs and the configuration recorded in the reported saved runs --------
# Resolve all file locations from this script's repository.
const repository = normpath(joinpath(@__DIR__, ".."))
# Source control and uniform-drag target files contain matched event initial conditions/forcing.
const data_directory = joinpath(repository, "Data", "thesis_v4")
# Every source-control fit begins from this same original source-pretrained checkpoint.
const pretrained_file = joinpath(repository, "Trained_models", "flux_correction_pretrained.bson")
# New source fits and their source/target forecast scores are saved here.
const output_directory = joinpath(repository, "results", "thesis_reproduction", "source_control")
# Observe h at four cells; the forecast still advances all 100 h cells and 101 velocity faces.
# h is water level [m]; velocity u [m/s] is not observed during fine-tuning.
const station_indices = [2, 4, 97, 99]
# Adam uses the original constant learning rate, moment decays and numerical epsilon.
const optimizer_settings = (learning_rate=3e-6, betas=(0.9, 0.999), epsilon=1e-8)
const training_settings = (
    # Train for 200 epochs, with at most four event-window pairs per optimiser update.
    epochs=200, batch_size=4,
    # Keep 128-step non-overlapping windows and the original list of available leads.
    rollout_horizons=[16, 32, 64, 128], window_stride=128,
    # Only lead 128 contributes to training loss, scaled by its pretrained training MSE.
    loss_mode=:endpoint, endpoint_horizon=128, normalize_loss=true,
    # No expanding lead schedule is used for this control.
    curriculum_enabled=false, curriculum_boundaries=[0.25, 0.375, 0.5],
    # Differentiate all 128 transitions. gradient_steps=1 has no effect while detachment is false.
    detach_gradients=false, gradient_steps=1, native_one_step=false,
    # Validate every five epochs with no early stopping; refresh states at each epoch and,
    # if reached, every 25 updates within it.
    validation_interval=5, patience=0, cache_refresh_interval=25,
)
# This record is passed to the shared fitter and saved unchanged in each result.jld2.
const configuration = (
    experiment=5, name="source_temporal_controls", output_directory=output_directory,
    # control contains source dynamics without the additional target-domain linear drag.
    dataset_file=joinpath(data_directory, "control.jld2"), pretrained_file=pretrained_file,
    # Use sparse water levels every 16 steps over the first 1,024 transitions.
    measurement_gap=16, training_steps=1024, stations=station_indices,
    # Observations are clean; retain the original independent noise-seed offsets as metadata.
    observation_noise_std=0.0, training_noise_seed_offset=710000,
    validation_noise_seed_offset=720000, optimizer_seed_offset=10000,
    # Full fine-tuning updates every network parameter and needs no LoRA ranks.
    optimizer=optimizer_settings, adaptation=(method=:full, ranks=nothing),
    training=training_settings,
)

# Defer loading numerical packages and model/data functions until execution is requested.
if "--run" in ARGS
    include(joinpath(@__DIR__, "training_checkpoint_io.jl"))
end

function main()
    # 2. Train on no-extra-drag data only -------------------------------------
    # Restore the starting weights, then load complete source and target event trajectories.
    pretrained_model = Surge.load_model(pretrained_file)
    # Each loader also returns the saved-state time interval in seconds.
    source_dataset, time_step = ConservativePilot.load_dataset(configuration.dataset_file)
    target_dataset, target_time_step = ConservativePilot.load_dataset(joinpath(data_directory, "uniform.jld2"))
    # Recreate the main campaign's paired train/validation/test memberships and optimisation seeds.
    # Five folds and three repeats produce 15 fits, with two training and four validation events each.
    event_splits = Surge.make_event_splits(18;
        fold_count=5, folds=collect(1:5), repeats=collect(1:3),
        training_event_count=2, validation_event_count=4,
        selection_seed=42000, training_seed=420000)
    # Record the runner, shared implementations and pinned package environment for provenance.
    source_files = [@__FILE__, joinpath(@__DIR__, "training_checkpoint_io.jl"),
        joinpath(repository, "thesis_experiments", "shared", "dataset_pretraining_utils.jl"),
        joinpath(repository, "Starting_networks", "Project.toml"),
        joinpath(repository, "Starting_networks", "Manifest.toml")]
    # Include every Julia implementation file under src in a stable order.
    append!(source_files, sort(filter(path -> endswith(path, ".jl"),
        readdir(joinpath(repository, "src"); join=true))))
    # SHA-256 digests identify the exact code, pretrained weights and source training ensemble.
    signature = (source_hashes=[bytes2hex(sha256(read(path))) for path in source_files],
        checkpoint=bytes2hex(sha256(read(pretrained_file))),
        data=bytes2hex(sha256(read(configuration.dataset_file))))
    # Create the output directory and save campaign metadata only on its first invocation.
    mkpath(output_directory)
    manifest_path = joinpath(output_directory, "manifest.jld2")
    if !isfile(manifest_path)
        # Preserve the original manifest keys, including settings, paths, date and Julia version.
        JLD2.jldsave(manifest_path; signature=signature, training=training_settings,
            optimizer=optimizer_settings, pretrained=pretrained_file,
            dataset=configuration.dataset_file, created=string(now()), julia_version=string(VERSION))
    end

    # Keep one index record per fit for later forecast evaluation and timing summaries.
    jobs = NamedTuple[]
    # Repeat source-only adaptation for every matched fold/repeat split.
    for event_split in event_splits
        # This directory names the objective and identifies its fold/repeat training fit.
        directory = joinpath(output_directory, "runs", "long128",
            "fold_$(event_split.fold)", "repeat_$(event_split.repeat)")
        # Train/resume using sparse source observations, or load an already completed fit.
        # The returned result records the best source-validation checkpoint and training times.
        result = fit_timed_run(pretrained_model, source_dataset, event_split, configuration, directory)
        # Keep weights in BSON; the index stores only run identity and selection/timing metadata.
        push!(jobs, (objective=:long128, fold=event_split.fold, repeat=event_split.repeat,
            directory=directory, best_epoch=result.best_epoch,
            best_validation_loss=result.best_validation_loss,
            training_seconds=result.training_seconds, preparation_seconds=result.preparation_seconds))
        # Update the human-readable index after each completed fit.
        Surge.save_table(joinpath(output_directory, "run_index.tsv"), jobs)
    end

    # 3. Source forecasts: retain the original diagnostics and table schemas --
    # Collect held-out source-event errors for adapted and frozen pretrained models.
    metrics = NamedTuple[]
    for job in jobs
        # Reuse existing forecast diagnostics when this run was evaluated previously.
        path = joinpath(job.directory, "diagnostics", "report_metrics.jld2")
        if isfile(path)
            report = JLD2.load(path, "result")
        else
            # Run continuous 72-hour forecasts from the full initial states, without observation updates.
            # Return full-domain/observed/interior h errors, velocity errors, bias and volume diagnostics.
            # Cells 9:92 define the interior; the hour boundaries define the four reporting intervals.
            report = ExperimentDiagnostics.diagnose_saved_run(job.directory, source_dataset;
                pretrained_model=pretrained_model, forecast_hours=72.0,
                interior_indices=collect(9:92), interval_boundaries=[0.0, 12.0, 24.0, 48.0, 72.0])
        end
        # Add objective and fold/repeat labels to each event/model/interval diagnostic row.
        for row in report.summaries
            push!(metrics, (objective=job.objective, fold=job.fold, repeat=job.repeat, row...))
        end
    end
    # Save individual repeat scores before calculating averages.
    Surge.save_table(joinpath(output_directory, "event_metrics.tsv"), metrics)
    # First group repeated fits of the same event, model and forecast interval.
    groups = Dict{Any, Vector{NamedTuple}}()
    for row in metrics
        # Omitting repeat from the key collects all three predictions for that held-out event.
        key = (row.objective, row.model, row.event, row.from_hours, row.to_hours)
        # Create a row list on first use and append this repeat to it.
        push!(get!(groups, key, NamedTuple[]), row)
    end
    # Reduce each event's three repeat scores to one MSE record, keeping events equally weighted later.
    events = NamedTuple[]
    for (key, rows) in groups
        # Unpack the event identity and interval endpoints represented by this group.
        objective, model, event, lower, upper = key
        # Average squared h/u errors, anomaly error and squared bias magnitude across repeats.
        push!(events, (objective=objective, model=model, event=event,
            from_hours=lower, to_hours=upper,
            mse_h=mean(row.mse_h for row in rows), mse_u=mean(row.mse_u for row in rows),
            anomaly_mse=mean(row.anomaly_mse for row in rows),
            bias_squared=mean(row.bias_rmse^2 for row in rows)))
    end
    # Sort rows for stable table ordering, then save the event-level repeat averages.
    sort!(events; by=row -> (string(row.objective), string(row.model), row.event, row.from_hours))
    Surge.save_table(joinpath(output_directory, "event_means.tsv"), events)

    # Build one condition summary for each model and forecast interval.
    summaries = NamedTuple[]
    # Interval start times identify 0-12, 12-24, 24-48 and 48-72 hour scores.
    for lower in (0.0, 12.0, 24.0, 48.0)
        # The frozen source-model errors provide the reference for reductions and event wins.
        baseline = filter(row -> row.model == :pretrained && row.from_hours == lower, events)
        # Give each of the 18 held-out events the same weight, despite unequal test-fold sizes.
        baseline_mse_h = mean(row.mse_h for row in baseline)
        baseline_mse_u = mean(row.mse_u for row in baseline)
        # Summarise the adapted and frozen models separately over these same events.
        for model in (:adapted, :pretrained)
            # Filter to the current model/interval, then average its event MSEs.
            rows = filter(row -> row.model == model && row.from_hours == lower, events)
            mse_h = mean(row.mse_h for row in rows)
            mse_u = mean(row.mse_u for row in rows)
            # Compare each event only with its matched frozen forecast; count strictly lower h MSEs.
            improved = count(row -> row.mse_h < only(filter(reference -> reference.event == row.event, baseline)).mse_h, rows)
            # Frozen forecasts incur no adaptation cost; adapted timing averages the 15 distinct fits.
            training_seconds = 0.0
            if model == :adapted
                training_seconds = mean(job.training_seconds for job in jobs)
            end
            # Keep MSE in the saved table; take its square root only when reporting RMSE.
            # 100*(1 - model MSE / frozen MSE) is percentage MSE reduction, not RMSE reduction.
            push!(summaries, (objective=:long128, model=model,
                from_hours=lower, to_hours=first(rows).to_hours, events=length(rows),
                mse_h=mse_h, mse_u=mse_u, anomaly_mse=mean(row.anomaly_mse for row in rows),
                bias_squared=mean(row.bias_squared for row in rows),
                reduction_h_percent=100 * (1 - mse_h / baseline_mse_h),
                reduction_u_percent=100 * (1 - mse_u / baseline_mse_u),
                events_improved_h=improved, training_seconds=training_seconds))
        end
    end
    # Export the original source-control summary columns for downstream analysis.
    Surge.save_table(joinpath(output_directory, "summary.tsv"), summaries)

    # 4. Target forecasts: use the original target-evaluation TSV columns -----
    # Target data never enter training, normalisation or checkpoint selection.
    # Reuse the same four reporting intervals for the independent target-domain evaluation.
    boundaries = [0.0, 12.0, 24.0, 48.0, 72.0]
    # Convert the 72-hour duration into a number of saved native forecast transitions.
    forecast_steps = ConservativePilot.steps_until(target_dataset, 72.0)
    # These post-initial-state timestamps align with the prediction/target arrays.
    forecast_times = target_dataset.times[1:forecast_steps]
    # Frozen predictions do not depend on fit repeat, so cache their interval scores by event number.
    baseline_scores = Dict{Int, Any}()
    # Gather the original target-evaluator columns and save them separately from source scores.
    target_rows = NamedTuple[]
    target_output = joinpath(output_directory, "target_evaluation")
    # Each source-trained model is now tested on the target events with the same held-out IDs.
    for job in jobs
        # Read test-event membership and restore the source-validation-selected model weights.
        result = JLD2.load(joinpath(job.directory, "result.jld2"), "result")
        # effective_model prepares ordinary convolution kernels for efficient forecast evaluation.
        model = Surge.effective_model(Surge.load_model(joinpath(job.directory, "best_model.bson")))
        for event_index in result.split.test
            # Include the known initial state plus one reference state per forecast transition.
            event = target_dataset.solution[event_index][1:forecast_steps + 1]
            # Compare unchanged pretrained weights with this source-fine-tuned model on identical inputs.
            for (label, forecast_model) in ((:pretrained, Surge.effective_model(pretrained_model)), (:adapted, model))
                # Reuse the frozen scores after the first repeat of this event has calculated them.
                if label == :pretrained && haskey(baseline_scores, event_index)
                    scores = baseline_scores[event_index]
                else
                    # Forecast autoregressively with prescribed forcing; later true states are evaluation targets only.
                    forecast = Surge.forecast_event(forecast_model, event, target_dataset.params)
                    # Compute each time step's errors using all 100 h cells and interior cells 9:92.
                    series = ExperimentDiagnostics.error_series(forecast.prediction, forecast.target,
                        first(event).h, target_dataset.params, collect(1:100), collect(9:92))
                    # Average the time-step error series over each specified reporting interval.
                    scores = ExperimentDiagnostics.summarize_error_series(series, forecast_times, boundaries)
                    # Save the unchanged model's scores for subsequent repeat evaluations of this event.
                    if label == :pretrained
                        baseline_scores[event_index] = scores
                    end
                end
                # Keep one row per model, event, repeat and interval; locations counts observations used in fitting.
                for score in scores
                    push!(target_rows, (locations=4, objective=:long128, fold=job.fold,
                        repeat=job.repeat, model=label, event=event_index,
                        from_hours=score.from_hours, to_hours=score.to_hours,
                        mse_h=score.mse_h, mse_u=score.mse_u))
                end
            end
        end
        # Save accumulated scores after each fit so finished evaluations remain available on interruption.
        Surge.save_table(joinpath(target_output, "event_metrics.tsv"), target_rows)
    end
    # Write a short text record of target inputs and the source-trained models being evaluated.
    # The do block closes the file after writing its provenance lines.
    open(joinpath(target_output, "provenance.txt"), "w") do output
        println(output, "Target dataset: $(joinpath(data_directory, "uniform.jld2"))")
        println(output, "Pretrained: $pretrained_file\nSparse: $output_directory")
        println(output, "Forecast: 72 h; no target training; native dt=$target_time_step seconds")
    end
end

# Run adaptation/evaluation only with --run; otherwise describe the planned control.
if "--run" in ARGS
    main()
else
    println("No-shift control: 15 source-domain long-128 fits with four gauges.")
    println("72-hour source and target evaluation. Target-trained comparator comes from runner 01.")
    println("Output: $output_directory\nAdd --run to execute.")
end
