module SourceControlAnalysis

using ..Surge, ..ExperimentDiagnostics
using JLD2, Statistics

function summarize_source_control(metrics, jobs)
    # Average repeats for each source event first.
    columns = (:objective, :model, :event, :from_hours, :to_hours,
        :mse_h, :mse_u, :anomaly_mse, :bias_squared)
    events = [NamedTuple{columns}(Tuple(getproperty(row, key) for key in columns))
        for row in ExperimentDiagnostics.average_event_errors(metrics;
            group_by=(:objective, :model, :event, :from_hours, :to_hours))]

    sort!(events; by=row -> (string(row.objective), string(row.model), row.event, row.from_hours))
    summaries = NamedTuple[]

    # Compare each interval with the pretrained model on the same events.
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

    return events, summaries
end

function evaluate_transfer(jobs, target_dataset, pretrained_model, target_output;
        forecast_hours, boundaries, stations, interior_indices)
    # Cache baseline scores and save transfer results as they are calculated.
    forecast_steps = searchsortedlast(target_dataset.times, forecast_hours * 3600 + 1e-6)
    forecast_times = target_dataset.times[1:forecast_steps]
    baseline_scores = Dict{Int, Any}()
    target_rows = NamedTuple[]
    for job in jobs
        result = JLD2.load(joinpath(job.directory, "result.jld2"), "result")
        # Build each saved model's effective kernels once and test matching target events.
        model = Surge.effective_model(Surge.load_model(joinpath(job.directory, "best_model.bson")))
        for event_index in result.split.test
            event = target_dataset.solution[event_index][1:forecast_steps + 1]
            for (label, forecast_model) in ((:pretrained, Surge.effective_model(pretrained_model)), (:adapted, model))
                # The pretrained forecast is the same across repeats. Cache its interval scores.
                if label == :pretrained && haskey(baseline_scores, event_index)
                    scores = baseline_scores[event_index]
                else
                    forecast = Surge.forecast_event(forecast_model, event, target_dataset.params)
                    # Use all water-level cells and the same forecast intervals.
                    series = ExperimentDiagnostics.error_series(forecast.prediction, forecast.target,
                        first(event).h, target_dataset.params, stations, interior_indices)
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
        # Save transfer scores after evaluating each fitted model.
        Surge.save_table(joinpath(target_output, "event_metrics.tsv"), target_rows)
    end

    return target_rows
end

function summarize_transfer(target_rows)
    # Report target MSE and percentage reductions from the pretrained model.
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
    return summary
end

end
