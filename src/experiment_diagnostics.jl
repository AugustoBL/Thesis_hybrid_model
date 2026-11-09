module ExperimentDiagnostics

using ..Surge
using JLD2, Statistics

include("diagnostic_aggregation.jl")

"""
Compute spatial error and volume diagnostics at each forecast step.

Args:
    prediction: Forecast array `(N + 1, 2, steps)` with padded h [m] and u [m/s].
    truth: Reference array matching prediction.
    initial_water_level: Initial physical cell levels `(N,)` [m], excluding padding.
    parameters: Domain width W and cell spacing dx [m].
    stations: Observed water-level cell indices.
    interior_indices: Interior water-level cell indices.

Returns:
    One record per step: h errors [m^2], u MSE [m^2/s^2], bias/amplitude error [m],
    and predicted/reference volume drift from the initial state [m^3].
"""
function error_series(prediction, truth, initial_water_level, parameters, stations, interior_indices)
    water_level_count = length(initial_water_level)
    # A change in mean water level times domain area gives a volume change.
    domain_area = parameters.W * parameters.dx * water_level_count
    initial_mean = mean(initial_water_level)
    rows = NamedTuple[]
    for step in axes(prediction, 3)
        # Exclude the padded h row and use Float64 for spatial diagnostics.
        predicted_level = Float64.(prediction[1:water_level_count, 1, step])
        reference_level = Float64.(truth[1:water_level_count, 1, step])
        level_error = predicted_level .- reference_level
        mean_bias = mean(level_error)
        # Remove each profile's own mean to isolate its spatial variation.
        predicted_anomaly = predicted_level .- mean(predicted_level)
        reference_anomaly = reference_level .- mean(reference_level)

        # Compare spatial RMS amplitudes; a positive value means excess predicted amplitude.
        amplitude_error = sqrt(mean(abs2, predicted_anomaly)) - sqrt(mean(abs2, reference_anomaly))
        # Keep full-domain, observed-cell, and interior-cell errors separate.
        push!(rows, (step=step, mse_h=mean(abs2, level_error),

            mse_u=mean(abs2, Float64.(prediction[:, 2, step] .- truth[:, 2, step])),
            mean_bias=mean_bias, bias_squared=mean_bias^2,
            anomaly_mse=mean(abs2, predicted_anomaly .- reference_anomaly),
            mse_observed=mean(abs2, level_error[stations]),
            mse_interior=mean(abs2, level_error[interior_indices]),
            amplitude_error=amplitude_error,
            volume_drift=domain_area * (mean(predicted_level) - initial_mean),
            reference_volume_drift=domain_area * (mean(reference_level) - initial_mean)))
    end
    return rows
end

"""
Summarize diagnostic series over forecast intervals.

Args:
    series: Per-step records from error_series.
    times: Elapsed prediction times `(steps,)` [s], one per series record.
    interval_boundaries: Ordered forecast boundaries [hours]; intervals are open-left and closed-right.

Returns:
    Interval records with mean errors, bias/amplitude RMSE, and endpoint volume drift [m^3].
"""
function summarize_error_series(series, times, interval_boundaries)
    summaries = NamedTuple[]
    for interval in 1:length(interval_boundaries)-1
        start_hour = interval_boundaries[interval]
        end_hour = interval_boundaries[interval+1]

        # Use (start, end] so adjacent intervals never count the same sample twice.
        indices = findall(time -> start_hour * 3600 < time <= end_hour * 3600, times)
        if isempty(indices)
            continue
        end
        rows = series[indices]
        # Average squared errors before taking RMS; volume drift uses the final sampled step.
        push!(summaries, (from_hours=start_hour, to_hours=end_hour,
            mse_h=mean(row.mse_h for row in rows), mse_u=mean(row.mse_u for row in rows),
            mean_bias=mean(row.mean_bias for row in rows),

            bias_rmse=sqrt(mean(row.bias_squared for row in rows)),
            anomaly_mse=mean(row.anomaly_mse for row in rows),
            mse_observed=mean(row.mse_observed for row in rows),
            mse_interior=mean(row.mse_interior for row in rows),
            amplitude_rmse=sqrt(mean(row.amplitude_error^2 for row in rows)),
            volume_drift_end=last(rows).volume_drift))
    end
    return summaries
end

"""
Evaluate supplied models on held-out events without writing files.

Args:
    models: Iterable of label => model pairs.
    dataset: Event trajectories including initial states, parameters, and elapsed prediction times [s].
    split: Record containing test-event indices.
    stations: Observed water-level cell indices.
    forecast_hours: Maximum forecast duration [hours].
    interior_indices: Water-level cells used for interior errors.
    interval_boundaries: Forecast interval boundaries [hours].
    keep_series: Retain per-step diagnostic records as well as interval summaries.
    include_finite: Add a finite_rollout field to summaries; nonfinite predictions always raise an error.

Returns:
    Named tuple of retained model/event series, interval summaries, and elapsed times [s].
"""
function evaluate_models(models, dataset, split, stations; forecast_hours=72.0,
                         interior_indices=collect(9:92),
                         interval_boundaries=[0.0, 12.0, 24.0, 48.0, 72.0],
                         keep_series=false, include_finite=false)
    # Select all sampled steps within the requested elapsed forecast duration.
    step_count = searchsortedlast(dataset.times, forecast_hours * 3600)
    times = dataset.times[1:step_count]
    series_records = NamedTuple[]
    summaries = NamedTuple[]
    for (label, model) in models, event_index in split.test
        # Include one extra state to initialize the forecast before its first prediction.
        event = dataset.solution[event_index][1:step_count+1]
        # Reuse materialized LoRA kernels throughout this event's autoregressive rollout.
        forecast = Surge.forecast_event(Surge.effective_model(model), event, dataset.params)
        finite = all(isfinite, forecast.prediction)
        # Reject unstable rollouts before calculating aggregate errors.
        finite || error("Nonfinite forecast: $label, event $event_index")
        series = error_series(forecast.prediction, forecast.target, first(event).h,
            dataset.params, stations, interior_indices)
        # Per-step series are optional to keep retained output small.
        keep_series && push!(series_records, (model=label, event=event_index, series=series))
        identity = include_finite ? (model=label, event=event_index, finite_rollout=finite) :
            (model=label, event=event_index)
        # Attach model/event identity to every interval summary.
        for summary in summarize_error_series(series, times, interval_boundaries)
            push!(summaries, (; identity..., summary...))
        end
    end
    return (series=series_records, summaries=summaries, times=times)
end

"""
Compare a saved adapted model with its pretrained baseline and save diagnostics.

Args:
    run_directory: Directory containing result.jld2 and best_model.bson.
    dataset: Event trajectories, parameters, and elapsed prediction times [s].
    pretrained_model: Baseline surrogate with matching geometry.
    forecast_hours: Maximum forecast duration [hours].
    interior_indices: Water-level cells used for interior errors.
    interval_boundaries: Ordered interval boundaries [hours].
    save_series: Save and return per-step records; interval summaries are always saved.

Returns:
    Evaluation records and times; writes diagnostics/metrics.tsv and optional model/event TSVs.
"""
function diagnose_saved_run(run_directory, dataset; pretrained_model, forecast_hours,
                            interior_indices, interval_boundaries, save_series=true)
    # Recover the held-out event split and the validation-selected model.
    saved = JLD2.load(joinpath(run_directory, "result.jld2"), "result")
    adapted_model = Surge.load_model(joinpath(run_directory, "best_model.bson"))
    # Evaluate both models on identical test events, stations, and forecast intervals.
    result = evaluate_models((:adapted => adapted_model, :pretrained => pretrained_model),
        dataset, saved.split, saved.configuration.stations; forecast_hours, interior_indices,
        interval_boundaries, keep_series=save_series)
    output_directory = joinpath(run_directory, "diagnostics")
    if save_series
        # Each file contains one model/event trajectory of diagnostic measurements.
        for record in result.series
            Surge.save_table(joinpath(output_directory, "$(record.model)_event_$(record.event).tsv"), record.series)
        end
    end
    # Interval summaries are retained even when per-step series are omitted.
    Surge.save_table(joinpath(output_directory, "metrics.tsv"), result.summaries)
    return result
end


end
