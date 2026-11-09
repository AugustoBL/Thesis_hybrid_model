module SparseCampaign

using ..Surge, ..ExperimentDiagnostics

function run_directory(output_directory, method, objective, profile, layout, noise_std, split)
    noise_cm = round(Int, noise_std * 100)
    name = "$(method)__$(objective)__$(profile)__$(layout)__noise$(noise_cm)cm"
    # Leave the stage out of the path so identical conditions reuse one saved fit.
    return joinpath(output_directory, "runs", name,
        "fold_$(split.fold)", "repeat_$(split.repeat)")
end

function run_condition!(jobs, pretrained_model, dataset, settings,
        stage, objective, profile, layout, split)
    method = settings.adaptation.method
    noise_std = settings.observation_noise_std
    output_directory = settings.output_directory
    directory = run_directory(output_directory, method, objective, profile, layout, noise_std, split)
    println("Stage $stage: $directory")
    # Train or load this condition and update the experiment index.
    result = Surge.fit_run(pretrained_model, dataset, split, settings, directory)
    push!(jobs, (stage=stage, method=method, temporal=objective, profile=profile,
        layout=layout, noise_std=noise_std, fold=split.fold, repeat=split.repeat,
        directory=directory, best_validation_loss=result.best_validation_loss,
        best_epoch=result.best_epoch,
        training_seconds=result.training_seconds, preparation_seconds=result.preparation_seconds))
    Surge.save_table(joinpath(output_directory, "run_index.tsv"), jobs)
    return result.best_validation_loss
end

function evaluate_runs!(jobs, pretrained_model, datasets; forecast_hours, interior_indices, interval_boundaries)
    # Evaluate each saved fit once.
    evaluated_directories = Set{String}()
    for job in jobs
        if job.directory in evaluated_directories
            continue
        end
        push!(evaluated_directories, job.directory)
        metrics_path = joinpath(job.directory, "diagnostics", "metrics.tsv")
        if !isfile(metrics_path)
            # Keep per-step series only for the chosen temporal comparisons.
            save_series = job.method == :full && job.profile == :uniform && job.layout == :both &&
                job.noise_std == 0.0 && job.temporal in (:balanced, :long, :one_step)
            ExperimentDiagnostics.diagnose_saved_run(job.directory, datasets[job.profile];
                pretrained_model=pretrained_model, forecast_hours=forecast_hours,
                interior_indices=interior_indices, interval_boundaries=interval_boundaries,
                save_series=save_series)
        end
    end

    return nothing
end

function collect_metrics(jobs)
    # Add stage labels to the interval results.
    metrics = NamedTuple[]
    for job in jobs
        rows = Surge.load_table(joinpath(job.directory, "diagnostics", "metrics.tsv"))
        if job.stage == 1
            condition = string(job.method)
        elseif job.stage == 2
            condition = string(job.temporal)
        elseif job.stage == 3
            condition = "$(job.profile)__$(job.layout)"
        else
            condition = "noise$(round(Int, 100 * job.noise_std))cm"
        end
        for row in rows
            push!(metrics, (stage=job.stage, condition=condition, fold=job.fold,
                repeat=job.repeat, training_seconds=job.training_seconds, row...))
        end
    end
    return metrics
end

function summarize_metrics(metrics)
    # Average repeats for each event first. Give each test event equal weight.
    columns = (:stage, :condition, :model, :from_hours, :to_hours, :events,
        :mse_h, :mse_u, :anomaly_mse, :bias_squared, :training_seconds)
    summaries = [NamedTuple{columns}(Tuple(getproperty(row, key) for key in columns))
        for row in ExperimentDiagnostics.summarize_run_diagnostics(metrics;
            group_by=(:stage, :condition, :model, :from_hours, :to_hours))]
    return summaries
end

end
