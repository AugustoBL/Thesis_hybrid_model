# Reported follow-ups only: two gauges (Section 6.4) and the shortened
# curriculum (Section 6.3). The other reviewer cases are not in this thesis.

# 1. Inputs, outputs and explicit training settings --------------------------
# Locate inputs relative to this script, rather than the current working directory.
const repository = normpath(joinpath(@__DIR__, ".."))
# Both follow-ups use the same 18 uniform-drag events as the main experiments.
const data_directory = joinpath(repository, "Data", "thesis_v4")
# Each new fit starts from the original source-trained model, not an adapted baseline.
const pretrained_file = joinpath(repository, "Trained_models", "flux_correction_pretrained.bson")
# Main-run selections determine the gauge comparison's fold-specific objective.
const baseline_directory = joinpath(repository, "results", "thesis_reproduction", "main")
# Keep the new fits, reference evaluations and comparison tables together.
const output_directory = joinpath(repository, "results", "thesis_reproduction", "followups")
# gauge_one means one gauge per boundary; curriculum_fast changes only the lead schedule.
# h denotes observed water level [m]; velocity u [m/s] evolves without direct supervision.
const reported_cases = [:gauge_one, :curriculum_fast]
# Adam uses a fixed learning rate, two moment-decay factors and a small denominator offset.
const optimizer_settings = (learning_rate=3e-6, betas=(0.9, 0.999), epsilon=1e-8)
const training_settings = (
    # A batch contains at most four event-window pairs; every fit has a 200-epoch budget.
    epochs=200, batch_size=4,
    # Leads and window spacing are native model steps; windows do not overlap.
    rollout_horizons=[16, 32, 64, 128], window_stride=128,
    # The default averages losses at all leads after scaling by pretrained training MSE.
    loss_mode=:balanced, endpoint_horizon=128, normalize_loss=true,
    # configuration below enables curriculum and replaces these stage boundaries when needed.
    curriculum_enabled=false, curriculum_boundaries=[0.25, 0.375, 0.5],
    # Differentiate through every active transition; the truncation fields remain unused here.
    detach_gradients=false, gradient_steps=16, native_one_step=false,
    # Select the best sparse validation MSE every five epochs, without early stopping.
    # Refresh window-start states each epoch; 25 is the additional within-epoch update interval.
    validation_interval=5, patience=0, cache_refresh_interval=25,
)

# Keep a plan-only launch free of dataset loading and training-package imports.
if "--run" in ARGS
    # Make the shared Surge and diagnostic modules available to the original runtime.
    include(joinpath(@__DIR__, "experiment_io.jl"))
    # Reuse the original timed fitter and interval evaluation/aggregation.
    include(joinpath(repository, "thesis_experiments", "shared", "reviewer_followup_runtime.jl"))
    # Calls below explicitly name the original fitting, evaluation and aggregation implementation.
    import .ReviewerFollowupRuntime
end

# Return one follow-up's complete fit settings for the given baseline temporal choice.
function configuration(case, temporal)
    # The default observes four h cells on the full 100-cell domain.
    stations = [2, 4, 97, 99]
    # Start from the same multi-step training settings before applying the case-specific change.
    settings = training_settings
    # For gauge density, change only the observed cells and retain the selected temporal loss.
    if case == :gauge_one
        stations = [2, 99] # one gauge per boundary, two in total
        # A fold that selected long-128 uses one endpoint instead of all four leads.
        if temporal == :long
            settings = merge(settings, (loss_mode=:endpoint,))
        end
    elseif case == :curriculum_fast
        # Boundaries at 10, 20 and 30 epochs: 10/10/10/170 allocation.
        settings = merge(settings, (curriculum_enabled=true,
            curriculum_boundaries=[0.05, 0.10, 0.15]))
    end
    # Preserve the original configuration fields expected by saved-run readers.
    return (
        experiment=6, name="reviewer_followups", case=case,
        output_directory=output_directory,
        dataset_file=joinpath(data_directory, "uniform.jld2"),
        # Supervise h every 16 transitions over the first 1,024 transitions of fitting events.
        pretrained_file=pretrained_file, measurement_gap=16, training_steps=1024,
        stations=stations, observation_noise_std=0.0,
        # Retain independent noise RNG offsets even though these two cases use clean observations.
        training_noise_seed_offset=710000, validation_noise_seed_offset=720000,
        # The split seed plus 10000 fixes optimisation order; full adaptation updates all parameters.
        optimizer_seed_offset=10000, optimizer=optimizer_settings,
        adaptation=(method=:full, ranks=nothing),
        # Retained metadata fields; neither reported case perturbs inputs.
        initial_rms=0.0, perturbation_seed=910000, perturbation_modes=4,
        # Pass the case-specific loss and lead schedule to the existing trainer.
        training=settings,
    )
end

function main()
    # 2. Load matched data, splits and the original fold-specific choices -----
    # Load complete trajectories and the native snapshot interval in seconds.
    dataset_file = joinpath(data_directory, "uniform.jld2")
    dataset, time_step = ConservativePilot.load_dataset(dataset_file)
    # Restore the common initial network used by every new fit and frozen comparison.
    pretrained_model = Surge.load_model(pretrained_file)
    # Recreate the same five folds, three repeats, event draws and seeds as the main runner.
    event_splits = Surge.make_event_splits(18;
        fold_count=5, folds=1:5, repeats=1:3,
        training_event_count=2, validation_event_count=4,
        selection_seed=42000, training_seed=420000)
    # Store one validation-selected objective name for each outer fold.
    selected_objectives = Dict{Int, Symbol}()
    for fold in 1:5
        # Selection tables are sorted best-first; line 1 is the header and line 2 is the winner.
        selection_path = joinpath(baseline_directory, "selections", "temporal_fold_$fold.tsv")
        # Read the first tab-separated field and convert its name to a Julia Symbol.
        selected_objectives[fold] = Symbol(first(split(readlines(selection_path)[2], '\t')))
    end

    # Retain the original manifest container, with provenance for this subset.
    # List this runner, its reused implementations and the package environment for fingerprinting.
    source_files = [@__FILE__, joinpath(@__DIR__, "experiment_io.jl"),
        joinpath(repository, "thesis_experiments", "shared", "reviewer_followup_runtime.jl"),
        joinpath(repository, "thesis_experiments", "shared", "conservative_pilot_utils.jl"),
        joinpath(repository, "Starting_networks", "Project.toml"),
        joinpath(repository, "Starting_networks", "Manifest.toml")]
    # Add all Julia files in src in a fixed order.
    append!(source_files, sort(filter(path -> endswith(path, ".jl"),
        readdir(joinpath(repository, "src"); join=true))))
    # The manifest retains every case/fold configuration and the files defining its baselines.
    settings = NamedTuple[]
    selection_paths = String[]
    baseline_files = String[]
    # Record both cases' scientific settings for all five folds, independent of repeat number.
    for case in reported_cases
        for fold in 1:5
            push!(settings, configuration(case, selected_objectives[fold]))
        end
    end
    # Fingerprint the main method and temporal selections on which these follow-ups depend.
    for fold in 1:5
        for stage in ("method", "temporal")
            push!(selection_paths, joinpath(baseline_directory, "selections", "$(stage)_fold_$fold.tsv"))
        end
    end
    # List the selected four/eight-gauge baselines and the fixed multi-step/curriculum references.
    for event_split in event_splits
        temporal = selected_objectives[event_split.fold]
        for (objective, layout) in ((temporal, :both), (temporal, :eight), (:balanced, :both), (:curriculum, :both))
            name = "full__$(objective)__uniform__$(layout)__noise0cm"
            directory = joinpath(baseline_directory, "runs", name,
                "fold_$(event_split.fold)", "repeat_$(event_split.repeat)")
            # Metadata identifies the split/settings; BSON supplies the selected model weights.
            push!(baseline_files, joinpath(directory, "result.jld2"))
            push!(baseline_files, joinpath(directory, "best_model.bson"))
        end
    end
    # Retain the main campaign's dataset fingerprint as a link to its provenance.
    baseline_signature = JLD2.load(joinpath(baseline_directory, "manifest.jld2"), "signature")
    # Hash exact inputs, source files, selection tables and unique reference files.
    # This metadata records a fresh reproduction; it does not modify the baseline campaign.
    signature = (checkpoint=bytes2hex(sha256(read(pretrained_file))),
        data=bytes2hex(sha256(read(dataset_file))),
        historical_data=first(baseline_signature.data), compatibility=(mode=:fresh_reproduction,),
        julia_version=string(VERSION), settings=settings,
        sources=[bytes2hex(sha256(read(path))) for path in source_files],
        selections=[bytes2hex(sha256(read(path))) for path in selection_paths],
        baselines=[bytes2hex(sha256(read(path))) for path in sort(unique(baseline_files))])
    # Create the output folder and save its manifest only when starting a new campaign.
    mkpath(output_directory)
    manifest_path = joinpath(output_directory, "manifest.jld2")
    if !isfile(manifest_path)
        # Two cases times five folds times three repeats gives 30 planned fits.
        JLD2.jldsave(manifest_path; signature=signature, created=string(now()), planned_runs=30)
    end

    # 3. Fit the two reported cases ------------------------------------------
    # jobs holds evaluation inputs; index_rows holds the readable training-result records.
    jobs = NamedTuple[]
    index_rows = NamedTuple[]
    # Fit each case on every matched fold/repeat split without new test-based selection.
    for case in reported_cases
        for event_split in event_splits
            # Choose the existing fold objective, then apply only this follow-up's change.
            temporal = selected_objectives[event_split.fold]
            settings = configuration(case, temporal)
            directory = joinpath(output_directory, "runs", string(case),
                "fold_$(event_split.fold)", "repeat_$(event_split.repeat)")
            # Reuse the original trainer orchestration, including resume state and best-epoch selection.
            result = ReviewerFollowupRuntime.fit(pretrained_model, dataset,
                event_split, settings, directory)
            # Retain the test membership and observed cells needed to evaluate this particular fit.
            push!(jobs, (case=case, split=event_split, stations=settings.stations, directory=directory))
            # Save selection/timing/update metadata without embedding model weights in the table.
            push!(index_rows, (case=case, fold=event_split.fold, repeat=event_split.repeat,
                best_epoch=result.best_epoch, training_seconds=result.training_seconds,
                updates=result.updates, directory=directory))
        end
    end
    # Keep the original filtered-run naming scheme for this exact subset of cases and splits.
    selection = "gauge_one-curriculum_fast_f1-2-3-4-5_r1-2-3"
    # Write the complete fit index as a tab-separated file.
    Surge.save_table(joinpath(output_directory, "indices", selection * ".tsv"), index_rows)

    # 4. Evaluate new fits and only the baselines needed for these comparisons -
    # Accumulate held-out event errors for new fits and their matched pretrained references.
    metrics = NamedTuple[]
    for job in jobs
        # Evaluate both the validation-selected adapted model and the unchanged source model.
        models = [:adapted => Surge.load_model(joinpath(job.directory, "best_model.bson")),
            :pretrained => pretrained_model]
        # Store interval metrics beside this fit instead of duplicating complete forecast arrays.
        evaluation_directory = joinpath(job.directory, "evaluation")
        path = joinpath(evaluation_directory, "interval_metrics.jld2")
        # A rerun can load finished interval evaluation without forecasting the same models again.
        if isfile(path)
            rows = JLD2.load(path, "rows")
        else
            # Forecast split.test events to 72 hours from full initial states with no observation updates.
            # Return per-event interval MSEs plus bias, anomaly, regional and volume diagnostics.
            rows = ReviewerFollowupRuntime.evaluate_models(models, dataset, job.split, job.stations)
            # Save a complete temporary JLD2 rows record, publish it, then export its readable TSV.
            mkpath(evaluation_directory)
            JLD2.jldsave(path * ".tmp"; rows=rows)
            mv(path * ".tmp", path; force=true)
            Surge.save_table(joinpath(evaluation_directory, "interval_metrics.tsv"), rows)
        end
        # Prefix each diagnostic row with the case and fold/repeat that produced it.
        for row in rows
            push!(metrics, (case=job.case, fold=job.split.fold, repeat=job.split.repeat, row...))
        end
    end

    # Legacy gauge labels count gauges per boundary: gauges_two has four total,
    # and gauges_four has eight total. The other references compare training schedules.
    reference_labels = [:gauges_two, :gauges_four, :multi128, :curriculum_original]
    # Evaluate existing selected models on the identical held-out event splits; never refit them.
    for label in reference_labels
        for event_split in event_splits
            # Gauge references retain the fold's chosen objective and default two-sided layout.
            temporal = selected_objectives[event_split.fold]
            layout = :both
            # Map each reference label to the original main-run objective/layout directory.
            if label == :gauges_four
                layout = :eight
            elseif label == :multi128
                temporal = :balanced
            elseif label == :curriculum_original
                temporal = :curriculum
            end
            # Reconstruct the scientific directory key instead of using old machine-specific paths.
            name = "full__$(temporal)__uniform__$(layout)__noise0cm"
            baseline_run = joinpath(baseline_directory, "runs", name,
                "fold_$(event_split.fold)", "repeat_$(event_split.repeat)")
            # Recover the baseline's original configuration, including its observation cells.
            result = JLD2.load(joinpath(baseline_run, "result.jld2"), "result")
            # Load selected baseline weights and pair them with the same frozen pretrained model.
            models = [:adapted => Surge.load_model(joinpath(baseline_run, "best_model.bson")),
                :pretrained => pretrained_model]
            # Write reference evaluations into this follow-up campaign, leaving main results untouched.
            evaluation_directory = joinpath(output_directory, "references", string(label),
                "fold_$(event_split.fold)", "repeat_$(event_split.repeat)")
            path = joinpath(evaluation_directory, "interval_metrics.jld2")
            # Reuse previous reference metrics or calculate them from the saved baseline model.
            if isfile(path)
                rows = JLD2.load(path, "rows")
            else
                # Use this baseline's observation layout for regional errors; future truth stays clean.
                rows = ReviewerFollowupRuntime.evaluate_models(models, dataset,
                    event_split, result.configuration.stations)
                # Publish JLD2 interval records and the same records as a readable table.
                mkpath(evaluation_directory)
                JLD2.jldsave(path * ".tmp"; rows=rows)
                mv(path * ".tmp", path; force=true)
                Surge.save_table(joinpath(evaluation_directory, "interval_metrics.tsv"), rows)
            end
            # Add these reference rows under their own labels for paired comparisons below.
            for row in rows
                push!(metrics, (case=label, fold=event_split.fold, repeat=event_split.repeat, row...))
            end
        end
    end

    # 5. Save exactly the original interval, event-mean and comparison schemas -
    # First average squared errors across repeats of an event, then weight events equally.
    # The returned summaries take square roots after averaging, giving h/u RMSEs.
    event_means, summaries = ReviewerFollowupRuntime.aggregate(metrics)
    # Keep the original partial_reports location used for a selected subset of reviewer cases.
    report_directory = joinpath(output_directory, "partial_reports", selection)
    # Save raw repeat records, event averages and final condition/interval summaries separately.
    Surge.save_table(joinpath(report_directory, "event_metrics.tsv"), metrics)
    Surge.save_table(joinpath(report_directory, "event_means.tsv"), event_means)
    Surge.save_table(joinpath(report_directory, "summary.tsv"), summaries)
    # Compare matched held-out events: RMSE change, percentage change and improved-event counts.
    comparisons = ReviewerFollowupRuntime.comparisons(event_means)
    # Export the original comparison columns for the thesis tables.
    Surge.save_table(joinpath(report_directory, "comparisons.tsv"), comparisons)
end

# Start the workflow only with --run; otherwise print a short description and output location.
if "--run" in ARGS
    main()
else
    println("Two reported follow-ups: two gauges and 10/10/10/170 curriculum.")
    println("30 fits; run 01_main_experiments.jl first, or edit baseline_directory.")
    println("Output: $output_directory")
    println("Add --run to execute.")
end
