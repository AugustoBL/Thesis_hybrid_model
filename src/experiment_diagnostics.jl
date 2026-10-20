module ExperimentDiagnostics

using ..Surge
using JLD2, Statistics

"""
Compute spatial error and volume diagnostics at each forecast step.

Args:
    prediction: Forecast array `(N + 1, 2, steps)` with padded h [m] and u [m/s].
    truth: Reference array matching prediction.
    initial_water_level: Initial N physical cell levels [m].
    parameters: Domain width W and cell spacing dx [m].
    stations: Observed water-level cell indices.
    interior_indices: Interior water-level cell indices.

Returns:
    Per-step records with h/u MSE, bias, anomaly and regional MSE, amplitude error, and volume drift.
"""
function error_series(prediction, truth, initial_water_level, parameters, stations, interior_indices)
    water_level_count = length(initial_water_level)
    # Uniform cells and constant width convert a domain-mean level change
    # into an integrated volume change.
    domain_area = parameters.W * parameters.dx * water_level_count
    initial_mean = mean(initial_water_level)
    rows = NamedTuple[]
    for step in axes(prediction, 3)
        # Discard padding and use Float64 for diagnostic level reductions.
        predicted_level = Float64.(prediction[1:water_level_count, 1, step])
        reference_level = Float64.(truth[1:water_level_count, 1, step])
        level_error = predicted_level .- reference_level
        mean_bias = mean(level_error)
        # Subtract each field's own spatial mean so shape error is separated
        # from a uniform water-level offset.
        predicted_anomaly = predicted_level .- mean(predicted_level)
        reference_anomaly = reference_level .- mean(reference_level)
        # Positive values indicate excessive spatial anomaly amplitude; this
        # diagnostic does not measure the timing or height of temporal peaks.
        amplitude_error = sqrt(mean(abs2, predicted_anomaly)) - sqrt(mean(abs2, reference_anomaly))
        push!(rows, (step=step, mse_h=mean(abs2, level_error),
            # Velocity error includes all faces. Its subtraction precedes Float64
            # conversion, unlike the level subtraction above.
            mse_u=mean(abs2, Float64.(prediction[:, 2, step] .- truth[:, 2, step])),
            mean_bias=mean_bias, bias_squared=mean_bias^2,
            anomaly_mse=mean(abs2, predicted_anomaly .- reference_anomaly),
            mse_observed=mean(abs2, level_error[stations]),
            mse_interior=mean(abs2, level_error[interior_indices]),
            amplitude_error=amplitude_error,
            # Measure volume relative to the initial state, not relative to the
            # reference at this time; report reference drift separately.
            volume_drift=domain_area * (mean(predicted_level) - initial_mean),
            reference_volume_drift=domain_area * (mean(reference_level) - initial_mean)))
    end
    return rows
end

"""
Summarize diagnostic series over forecast intervals.

Args:
    series: Per-step records from error_series.
    times: Matching prediction times [s].
    interval_boundaries: Ordered forecast boundaries [hours]; intervals are open-left and closed-right.

Returns:
    Interval records with mean errors, bias/amplitude RMSE, and endpoint volume drift [m^3].
"""
function summarize_error_series(series, times, interval_boundaries)
    summaries = NamedTuple[]
    for interval in 1:length(interval_boundaries)-1
        start_hour = interval_boundaries[interval]
        end_hour = interval_boundaries[interval+1]
        # Open-left/closed-right boundaries prevent a shared endpoint from
        # being counted in both neighbouring intervals.
        indices = findall(time -> start_hour * 3600 < time <= end_hour * 3600, times)
        if isempty(indices)
            continue
        end
        rows = series[indices]
        push!(summaries, (from_hours=start_hour, to_hours=end_hour,
            mse_h=mean(row.mse_h for row in rows), mse_u=mean(row.mse_u for row in rows),
            mean_bias=mean(row.mean_bias for row in rows),
            # Square before averaging so positive and negative biases cannot cancel
            # in the reported bias magnitude.
            bias_rmse=sqrt(mean(row.bias_squared for row in rows)),
            anomaly_mse=mean(row.anomaly_mse for row in rows),
            mse_observed=mean(row.mse_observed for row in rows),
            mse_interior=mean(row.mse_interior for row in rows),
            amplitude_rmse=sqrt(mean(row.amplitude_error^2 for row in rows)),
            # Volume is reported at the interval endpoint, rather than averaged
            # through the interval like MSE.
            volume_drift_end=last(rows).volume_drift))
    end
    return summaries
end

"""
Evaluate saved models and write forecast diagnostic series.

Args:
    run_directory: Directory containing result.jld2 and best_model.bson.
    dataset: Event trajectories, parameters, and prediction times [s].
    pretrained_model: Baseline surrogate.
    forecast_hours: Maximum forecast duration [hours].
    interior_indices: Water-level cells used for interior errors.
    interval_boundaries: Forecast interval boundaries [hours].

Returns:
    Named tuple of series, interval summaries, and times; saves diagnostic files.
"""
function diagnose_saved_run(run_directory, dataset; pretrained_model, forecast_hours,
                            interior_indices, interval_boundaries)
    saved = JLD2.load(joinpath(run_directory, "result.jld2"), "result")
    adapted_model = Surge.load_model(joinpath(run_directory, "best_model.bson"))
    step_count = searchsortedlast(dataset.times, forecast_hours * 3600)
    times = dataset.times[1:step_count]
    output_directory = joinpath(run_directory, "diagnostics")
    series_records = NamedTuple[]
    summary_rows = NamedTuple[]
    # Apply identical event membership, duration, and diagnostics to both
    # models for a paired comparison.
    for (label, model) in ((:adapted, adapted_model), (:pretrained, pretrained_model))
        for event_index in saved.split.test
            event = dataset.solution[event_index][1:step_count+1]
            forecast = Surge.forecast_event(Surge.effective_model(model), event, dataset.params)
            # Use the same original initial level for both volume-drift calculations.
            series = error_series(forecast.prediction, forecast.target, first(event).h,
                dataset.params, saved.configuration.stations, interior_indices)
            push!(series_records, (model=label, event=event_index, series=series))
            summaries = summarize_error_series(series, times, interval_boundaries)
            for summary in summaries
                push!(summary_rows, (model=label, event=event_index, summary...))
            end
            Surge.save_table(joinpath(output_directory, "$(label)_event_$(event_index).tsv"), series)
        end
    end
    # Keep readable interval tables alongside the structured series object
    # used by plotting routines.
    Surge.save_table(joinpath(output_directory, "metrics.tsv"), summary_rows)
    result = (series=series_records, summaries=summary_rows, times=times)
    JLD2.jldsave(joinpath(output_directory, "report_metrics.jld2"); result=result)
    return result
end

# Aggregation reads saved diagnostic summaries; it never forecasts or fits a model.
include("diagnostic_aggregation.jl")

end
