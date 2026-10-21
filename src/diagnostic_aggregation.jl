# Included inside ExperimentDiagnostics.

"""Collect retained interval TSV diagnostics with run metadata and measured training time."""
function collect_run_diagnostics(runs; output_path=nothing)
    records = NamedTuple[]
    for run in runs
        saved = JLD2.load(joinpath(run.directory, "result.jld2"), "result")
        metadata = get(run, :metadata, NamedTuple())
        for row in Surge.load_table(joinpath(run.directory, "diagnostics", "metrics.tsv"))
            push!(records, (; metadata..., fold=saved.split.fold, repeat=saved.split.repeat,
                directory=run.directory, training_seconds=get(saved, :training_seconds, missing), row...))
        end
    end
    isnothing(output_path) || Surge.save_table(output_path, records)
    return records
end

"""Average repeats within each event, keeping scientific conditions and intervals separate."""
function average_event_errors(records; group_by)
    fields = Tuple(group_by)
    groups = Dict{Any, Vector{NamedTuple}}()
    for row in records
        key = NamedTuple{fields}(Tuple(getproperty(row, field) for field in fields))
        push!(get!(groups, key, NamedTuple[]), row)
    end
    events = NamedTuple[]
    for (key, rows) in groups
        length(unique((row.fold, row.repeat) for row in rows)) == length(rows) ||
            error("Duplicate event/repeat: $key")
        length(unique(row.fold for row in rows)) == 1 || error("Event appears in multiple test folds: $key")
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

"""Average event MSEs equally; training cost averages distinct fits, with zero for pretrained models."""
function summarize_run_diagnostics(records; group_by, output_path=nothing)
    fields = Tuple(group_by)
    events = average_event_errors(records; group_by=(fields..., :event))
    groups = Dict{Any, Vector{NamedTuple}}()
    for event in events
        key = NamedTuple{fields}(Tuple(getproperty(event, field) for field in fields))
        push!(get!(groups, key, NamedTuple[]), event)
    end
    summaries = NamedTuple[]
    for key in sort!(collect(keys(groups)); by=repr)
        rows = groups[key]
        fits = Dict((row.fold, row.repeat) => get(row, :training_seconds, missing) for row in records
            if all(getproperty(row, field) == getproperty(key, field) for field in fields))
        average = field -> mean(getproperty(row, field) for row in rows)
        training_seconds = key.model == :pretrained ? 0.0 :
            any(ismissing, values(fits)) ? missing : mean(collect(values(fits)))
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
