const repository = normpath(joinpath(@__DIR__, ".."))
const data_directory = joinpath(repository, "Data", "thesis_v4")
const pretrained_file = joinpath(repository, "Trained_models", "flux_correction_pretrained.bson")
const output_directory = joinpath(repository, "results", "experiments", "source_domain_control")
const station_indices = [2, 4, 97, 99]
const optimizer_settings = (learning_rate=3e-6, betas=(0.9, 0.999), epsilon=1e-8)
const training_settings = (
    epochs=200, batch_size=4,
    rollout_horizons=[16, 32, 64, 128], window_stride=128,
    loss_mode=:endpoint, endpoint_horizon=128, normalize_loss=true,
    curriculum_enabled=false, curriculum_boundaries=[0.25, 0.375, 0.5],
    detach_gradients=false, gradient_steps=1,
    validation_interval=5, patience=0, cache_refresh_interval=25,
)
const configuration = (
    experiment=5, name="source_temporal_controls", output_directory=output_directory,
    dataset_file=joinpath(data_directory, "control.jld2"), pretrained_file=pretrained_file,
    measurement_gap=16, training_steps=1024, stations=station_indices,
    observation_noise_std=0.0, training_noise_seed_offset=710000,
    validation_noise_seed_offset=720000, optimizer_seed_offset=10000,
    optimizer=optimizer_settings, adaptation=(method=:full, ranks=nothing),
    training=training_settings,
)

include(joinpath(@__DIR__, "experiment_dependencies.jl"))

function main()
    pretrained_model = Surge.load_model(pretrained_file)
    source_dataset = Surge.load_burgers_ensemble(basename(configuration.dataset_file);
        data_path=dirname(configuration.dataset_file), relative_times=true)
    target_dataset = Surge.load_burgers_ensemble(basename(joinpath(data_directory, "uniform.jld2"));
        data_path=dirname(joinpath(data_directory, "uniform.jld2")), relative_times=true)
    event_splits = Surge.make_event_splits(18;
        fold_count=5, folds=collect(1:5), repeats=collect(1:3),
        training_event_count=2, validation_event_count=4,
        selection_seed=42000, training_seed=420000)

    jobs = NamedTuple[]
    for event_split in event_splits
        directory = joinpath(output_directory, "runs", "long128",
            "fold_$(event_split.fold)", "repeat_$(event_split.repeat)")
        result = Surge.fit_run(pretrained_model, source_dataset, event_split, configuration, directory)
        push!(jobs, (objective=:long128, fold=event_split.fold, repeat=event_split.repeat,
            directory=directory, best_epoch=result.best_epoch,
            best_validation_loss=result.best_validation_loss,
            training_seconds=result.training_seconds, preparation_seconds=result.preparation_seconds))
        Surge.save_table(joinpath(output_directory, "run_index.tsv"), jobs)
    end

    metrics = NamedTuple[]
    for job in jobs
        path = joinpath(job.directory, "diagnostics", "metrics.tsv")
        if isfile(path)
            rows = Surge.load_table(path)
        else
            report = ExperimentDiagnostics.diagnose_saved_run(job.directory, source_dataset;
                pretrained_model=pretrained_model, forecast_hours=72.0,
                interior_indices=collect(9:92), interval_boundaries=[0.0, 12.0, 24.0, 48.0, 72.0],
                save_series=false)
            rows = report.summaries
        end
        for row in rows
            push!(metrics, (objective=job.objective, fold=job.fold, repeat=job.repeat, row...))
        end
    end
    Surge.save_table(joinpath(output_directory, "event_metrics.tsv"), metrics)
    columns = (:objective, :model, :event, :from_hours, :to_hours,
        :mse_h, :mse_u, :anomaly_mse, :bias_squared)
    events = [NamedTuple{columns}(Tuple(getproperty(row, key) for key in columns))
        for row in ExperimentDiagnostics.average_event_errors(metrics;
            group_by=(:objective, :model, :event, :from_hours, :to_hours))]

    sort!(events; by=row -> (string(row.objective), string(row.model), row.event, row.from_hours))
    Surge.save_table(joinpath(output_directory, "event_means.tsv"), events)

    summaries = NamedTuple[]

    for lower in (0.0, 12.0, 24.0, 48.0)

        baseline = filter(row -> row.model == :pretrained && row.from_hours == lower, events)
        baseline_mse_h = mean(row.mse_h for row in baseline)
        baseline_mse_u = mean(row.mse_u for row in baseline)
        for model in (:adapted, :pretrained)
            rows = filter(row -> row.model == model && row.from_hours == lower, events)
            mse_h = mean(row.mse_h for row in rows)
            mse_u = mean(row.mse_u for row in rows)
            improved = count(row -> row.mse_h < only(filter(reference -> reference.event == row.event, baseline)).mse_h, rows)
            training_seconds = 0.0
            if model == :adapted
                training_seconds = mean(job.training_seconds for job in jobs)
            end
            push!(summaries, (objective=:long128, model=model,
                from_hours=lower, to_hours=first(rows).to_hours, events=length(rows),
                mse_h=mse_h, mse_u=mse_u, anomaly_mse=mean(row.anomaly_mse for row in rows),
                bias_squared=mean(row.bias_squared for row in rows),
                reduction_h_percent=100 * (1 - mse_h / baseline_mse_h),
                reduction_u_percent=100 * (1 - mse_u / baseline_mse_u),
                events_improved_h=improved, training_seconds=training_seconds))
        end
    end

    Surge.save_table(joinpath(output_directory, "summary.tsv"), summaries)

    boundaries = [0.0, 12.0, 24.0, 48.0, 72.0]
    forecast_steps = searchsortedlast(target_dataset.times, 72.0 * 3600 + 1e-6)
    forecast_times = target_dataset.times[1:forecast_steps]
    baseline_scores = Dict{Int, Any}()
    target_rows = NamedTuple[]
    target_output = joinpath(repository, "results", "experiments", "source_to_target")
    for job in jobs
        result = JLD2.load(joinpath(job.directory, "result.jld2"), "result")
        model = Surge.effective_model(Surge.load_model(joinpath(job.directory, "best_model.bson")))
        for event_index in result.split.test
            event = target_dataset.solution[event_index][1:forecast_steps + 1]
            for (label, forecast_model) in ((:pretrained, Surge.effective_model(pretrained_model)), (:adapted, model))
                if label == :pretrained && haskey(baseline_scores, event_index)
                    scores = baseline_scores[event_index]
                else
                    forecast = Surge.forecast_event(forecast_model, event, target_dataset.params)
                    series = ExperimentDiagnostics.error_series(forecast.prediction, forecast.target,
                        first(event).h, target_dataset.params, collect(1:100), collect(9:92))
                    scores = ExperimentDiagnostics.summarize_error_series(series, forecast_times, boundaries)
                    if label == :pretrained
                        baseline_scores[event_index] = scores
                    end
                end
                for score in scores
                    push!(target_rows, (locations=4, objective=:long128, fold=job.fold,
                        repeat=job.repeat, model=label, event=event_index,
                        from_hours=score.from_hours, to_hours=score.to_hours,
                        mse_h=score.mse_h, mse_u=score.mse_u))
                end
            end
        end
        Surge.save_table(joinpath(target_output, "event_metrics.tsv"), target_rows)
    end

    summary = NamedTuple[]
    for lower in (0.0, 12.0, 24.0, 48.0)
        baseline = filter(row -> row.model == :pretrained && row.from_hours == lower, target_rows)
        baseline_h, baseline_u = mean(row.mse_h for row in baseline), mean(row.mse_u for row in baseline)
        for model in (:pretrained, :adapted)
            rows = filter(row -> row.model == model && row.from_hours == lower, target_rows)
            mse_h, mse_u = mean(row.mse_h for row in rows), mean(row.mse_u for row in rows)
            push!(summary, (locations=4, objective=:long128, model=model,
                from_hours=lower, to_hours=first(rows).to_hours, mse_h=mse_h, mse_u=mse_u,
                mse_h_reduction_percent=100 * (1 - mse_h / baseline_h),
                mse_u_reduction_percent=100 * (1 - mse_u / baseline_u)))
        end
    end
    Surge.save_table(joinpath(target_output, "summary.csv"), summary; delimiter=',')

end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
