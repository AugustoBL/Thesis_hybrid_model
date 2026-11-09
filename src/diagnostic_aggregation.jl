# Included inside ExperimentDiagnostics.

"""
Collect saved interval diagnostics and attach run metadata.

Args:
    runs: Records with a directory and optional metadata fields identifying experimental conditions.
    output_path: Optional destination TSV file path.

Returns:
    Interval records with condition metadata, fold, repeat, directory and training time [s].
    Training time is missing when absent from the saved result.
"""
function collect_run_diagnostics(runs; output_path=nothing)
    records = NamedTuple[]
    for run in runs
        saved = JLD2.load(joinpath(run.directory, "result.jld2"), "result")
        metadata = get(run, :metadata, NamedTuple())
        # Each row gives scores for one model, test event and forecast interval.
        for row in Surge.load_table(joinpath(run.directory, "diagnostics", "metrics.tsv"))
            push!(records, (; metadata..., fold=saved.split.fold, repeat=saved.split.repeat,
                directory=run.directory, training_seconds=get(saved, :training_seconds, missing), row...))
        end
    end
    isnothing(output_path) || Surge.save_table(output_path, records)
    return records
end

"""
Average repeated fits separately for each test event and experimental condition.

Args:
    records: Interval diagnostic records with fold/repeat identifiers.
    group_by: Fields identifying a condition, model, forecast interval and event. Exclude repeat.

Returns:
    Per-event records with equal-repeat averages, repeat counts and finite-rollout flags.
    Bias and amplitude RMS values are squared before averaging.
"""
function average_event_errors(records; group_by)
    fields = Tuple(group_by)
    groups = Dict{Any, Vector{NamedTuple}}()
    for row in records
        key = NamedTuple{fields}(Tuple(getproperty(row, field) for field in fields))
        push!(get!(groups, key, NamedTuple[]), row)
    end
    events = NamedTuple[]
    for (key, rows) in groups
        # Each event must occur once per fit and belong to one test fold.
        length(unique((row.fold, row.repeat) for row in rows)) == length(rows) ||
            error("Duplicate event/repeat: $key")
        length(unique(row.fold for row in rows)) == 1 || error("Event appears in multiple test folds: $key")
        # Average squared errors before taking RMS to keep the correct weighting.
        push!(events, (; key..., repeats=length(rows),
            finite_rollout=all(row -> get(row, :finite_rollout, true), rows),
            mse_h=mean(row.mse_h for row in rows), mse_u=mean(row.mse_u for row in rows),
            anomaly_mse=mean(row.anomaly_mse for row in rows),
            mse_observed=mean(row.mse_observed for row in rows),
            mse_interior=mean(row.mse_interior for row in rows),
            mean_bias=mean(row.mean_bias for row in rows),
            bias_squared=mean(row.bias_rmse^2 for row in rows),
            amplitude_mse=mean(row.amplitude_rmse^2 for row in rows),
            volume_drift_end=mean(row.volume_drift_end for row in rows)))
    end
    return events
end

"""
Summarize repeated-run diagnostics with equal weight for each test event.

Args:
    records: Interval diagnostics with event, fold, repeat and optional training_seconds fields.
    group_by: Condition, model and interval fields. Include model and exclude event/fold/repeat.
    output_path: Optional destination TSV file path.

Returns:
    Group summaries with event/fit counts, MSE/RMSE, bias, amplitude, volume drift [m^3]
    and mean training cost [s]. Pretrained cost is zero. Unavailable fitted costs remain missing.
"""
function summarize_run_diagnostics(records; group_by, output_path=nothing)
    fields = Tuple(group_by)
    # Average repeats first so events with more repeats do not receive extra weight.
    events = average_event_errors(records; group_by=(fields..., :event))
    groups = Dict{Any, Vector{NamedTuple}}()
    for event in events
        # Group events by condition and interval.
        key = NamedTuple{fields}(Tuple(getproperty(event, field) for field in fields))
        push!(get!(groups, key, NamedTuple[]), event)
    end
    summaries = NamedTuple[]
    # Sort group keys for reproducible output order.
    for key in sort!(collect(keys(groups)); by=repr)
        rows = groups[key]
        # Count each fit's training time once, even when it evaluated several test events.
        fits = Dict((row.fold, row.repeat) => get(row, :training_seconds, missing) for row in records
            if all(getproperty(row, field) == getproperty(key, field) for field in fields))
        average = field -> mean(getproperty(row, field) for row in rows)
        # The pretrained model has zero adaptation cost. Keep missing fit times as missing.
        training_seconds = key.model == :pretrained ? 0.0 :
            any(ismissing, values(fits)) ? missing : mean(collect(values(fits)))
        # Give events equal weight and take square roots after averaging squared errors.
        push!(summaries, (; key..., events=length(rows), fits=length(fits),
            repeats_min=minimum(row.repeats for row in rows), repeats_max=maximum(row.repeats for row in rows),
            mse_h=average(:mse_h), mse_u=average(:mse_u),
            rmse_h=sqrt(average(:mse_h)), rmse_u=sqrt(average(:mse_u)),
            anomaly_mse=average(:anomaly_mse), mse_observed=average(:mse_observed),
            mse_interior=average(:mse_interior), mean_bias=average(:mean_bias),
            bias_squared=average(:bias_squared), bias_rmse=sqrt(average(:bias_squared)),
            amplitude_mse=average(:amplitude_mse), amplitude_rmse=sqrt(average(:amplitude_mse)),
            volume_drift_end=average(:volume_drift_end), training_seconds=training_seconds,
            finite_rollout=all(row.finite_rollout for row in rows)))
    end
    isnothing(output_path) || Surge.save_table(output_path, summaries)
    return summaries
end
