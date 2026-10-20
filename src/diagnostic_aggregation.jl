# Included inside ExperimentDiagnostics, alongside the per-run diagnostic functions.

"""
    collect_run_diagnostics(runs; output_path=nothing)

Combine existing `diagnostics/report_metrics.jld2` interval summaries into records.
Each run supplies `directory` and may supply a `metadata` named tuple, for example
`(condition="full_long", stage=2)`. Fold/repeat come from its saved `result.jld2`.
Metadata cannot replace diagnostic fields, run identifiers, or measured timing.
`training_seconds` is `missing` when the saved fit did not measure it.

Returns rows and optionally writes a TSV table. Missing reports raise an error;
this function never regenerates forecasts. It retains MSE rather than converting
individual event errors to RMSE before aggregation.
"""
function collect_run_diagnostics(runs; output_path=nothing)
    records = NamedTuple[]
    for run in runs
        directory = abspath(run.directory)
        report_path = joinpath(directory, "diagnostics", "report_metrics.jld2")
        result_path = joinpath(directory, "result.jld2")
        isfile(report_path) || throw(ArgumentError("Missing saved diagnostics: $report_path"))
        isfile(result_path) || throw(ArgumentError("Missing saved fit record: $result_path"))
        saved = JLD2.load(result_path, "result")
        report = JLD2.load(report_path, "result")
        metadata = hasproperty(run, :metadata) ? run.metadata : NamedTuple()
        metadata isa NamedTuple || throw(ArgumentError("Run metadata must be a named tuple"))
        # Timing is historical wall-clock information, not a value inferred from errors.
        timing = hasproperty(saved, :training_seconds) ? saved.training_seconds : missing
        identity = (fold=saved.split.fold, repeat=saved.split.repeat,
            directory=directory, training_seconds=timing)
        for row in report.summaries
            # Reject collisions instead of silently changing a model label or error value.
            isempty(intersect(propertynames(metadata), (propertynames(identity)..., propertynames(row)...))) ||
                throw(ArgumentError("Metadata overlaps reserved run/diagnostic fields"))
            record = (; metadata..., identity..., row...)
            if !isempty(records) && propertynames(record) != propertynames(first(records))
                throw(ArgumentError("All runs must provide the same ordered metadata and diagnostic columns"))
            end
            push!(records, record)
        end
    end
    isempty(records) && throw(ArgumentError("No saved diagnostic summaries were supplied"))
    isnothing(output_path) || Surge.save_table(output_path, records)
    return records
end

"""
    summarize_run_diagnostics(records; group_by, output_path=nothing)

Average repeats within each event, then weight events equally within each group.
For example, `group_by=(:condition, :model, :from_hours, :to_hours)` keeps scientific
conditions, adapted/pretrained forecasts and lead intervals separate. Include all
fields that distinguish conditions; do not pool different scenarios accidentally.
`model`, `from_hours`, and `to_hours` are mandatory grouping fields.

Returns MSE and RMSE for h/u, anomaly and regional MSE, signed mean bias, bias RMS,
amplitude RMS and signed mean endpoint volume drift. RMSE is taken AFTER averaging
squared errors. No forecasts are averaged. Duplicate event/repeat records and
events occurring in multiple test folds within a group are rejected.

Training time averages distinct fits, not event rows; pretrained time is zero.
Missing adapted timings remain missing (no partial timing average). Repeat-count
minimum/maximum make unequal repeat coverage visible rather than silently hiding it.
Intervals are inherited from the saved reports and cannot be re-binned here.
"""
function summarize_run_diagnostics(records; group_by, output_path=nothing)
    fields = Tuple(group_by)
    all(field -> field isa Symbol, fields) || throw(ArgumentError("group_by must contain column symbols"))
    length(unique(fields)) == length(fields) || throw(ArgumentError("Duplicate grouping columns"))
    all(field -> field in fields, (:model, :from_hours, :to_hours)) ||
        throw(ArgumentError("Group separately by model, from_hours and to_hours"))
    any(field -> field in fields, (:event, :repeat)) &&
        throw(ArgumentError("Event and repeat are averaging axes, not grouping columns"))
    isempty(records) && throw(ArgumentError("No diagnostic records were supplied"))
    metrics = (:mse_h, :mse_u, :anomaly_mse, :mse_observed, :mse_interior,
        :mean_bias, :bias_rmse, :amplitude_rmse, :volume_drift_end)
    groups = Dict{Any,Vector{NamedTuple}}()
    for row in records
        required = (fields..., :fold, :repeat, :event, metrics...)
        all(field -> hasproperty(row, field), required) || throw(ArgumentError("Missing grouping or diagnostic column"))
        all(field -> getproperty(row, field) isa Real && isfinite(getproperty(row, field)), metrics) ||
            throw(ArgumentError("Diagnostic metrics must be finite real numbers"))
        all(field -> getproperty(row, field) >= 0, (:mse_h, :mse_u, :anomaly_mse,
            :mse_observed, :mse_interior, :bias_rmse, :amplitude_rmse)) ||
            throw(ArgumentError("Squared errors and RMS magnitudes must be nonnegative"))
        key = NamedTuple{fields}(Tuple(getproperty(row, field) for field in fields))
        push!(get!(groups, key, NamedTuple[]), row)
    end
    summaries = NamedTuple[]
    for key in sort!(collect(keys(groups)); by=repr)
        rows = groups[key]
        events = Dict{Any,Vector{NamedTuple}}()
        seen = Set{Any}()
        timings = Dict{Any,Any}()
        for row in rows
            identity = (row.fold, row.repeat, row.event)
            identity in seen && throw(ArgumentError("Duplicate event/repeat in group $key: $identity"))
            push!(seen, identity)
            push!(get!(events, row.event, NamedTuple[]), row)
            fit = (row.fold, row.repeat)
            timing = hasproperty(row, :training_seconds) ? row.training_seconds : missing
            if !ismissing(timing)
                timing isa Real && isfinite(timing) && timing >= 0 ||
                    throw(ArgumentError("Training time must be missing or finite and nonnegative"))
            end
            haskey(timings, fit) && !isequal(timings[fit], timing) &&
                throw(ArgumentError("Inconsistent timing for the same fold/repeat"))
            timings[fit] = timing
        end
        event_means = NamedTuple[]
        repeats_per_event = Int[]
        for event in sort!(collect(keys(events)); by=repr)
            repetitions = events[event]
            length(unique(row.fold for row in repetitions)) == 1 ||
                throw(ArgumentError("Event $event occurs in multiple test folds within group $key"))
            push!(repeats_per_event, length(repetitions))
            # First compute E_r[MSE_{event,r}]; differing repeat counts do not change event weights.
            push!(event_means, (
                mse_h=mean(row.mse_h for row in repetitions),
                mse_u=mean(row.mse_u for row in repetitions),
                anomaly_mse=mean(row.anomaly_mse for row in repetitions),
                mse_observed=mean(row.mse_observed for row in repetitions),
                mse_interior=mean(row.mse_interior for row in repetitions),
                mean_bias=mean(row.mean_bias for row in repetitions),
                # Square RMS magnitudes before averaging, so opposite biases cannot cancel their RMS.
                bias_squared=mean(row.bias_rmse^2 for row in repetitions),
                amplitude_mse=mean(row.amplitude_rmse^2 for row in repetitions),
                volume_drift_end=mean(row.volume_drift_end for row in repetitions)))
        end
        # Then compute E_event[E_repeat[MSE]]; every held-out event receives weight 1/N.
        average = field -> mean(getproperty(event, field) for event in event_means)
        # Deduplicate fits before timing averages; folds with more test events get no extra timing weight.
        training_seconds = string(key.model) == "pretrained" ? 0.0 :
            any(ismissing, values(timings)) ? missing : mean(collect(values(timings)))
        statistics = (events=length(event_means), fits=length(timings),
            repeats_min=minimum(repeats_per_event), repeats_max=maximum(repeats_per_event),
            mse_h=average(:mse_h), mse_u=average(:mse_u),
            rmse_h=sqrt(average(:mse_h)), rmse_u=sqrt(average(:mse_u)),
            anomaly_mse=average(:anomaly_mse), mse_observed=average(:mse_observed),
            mse_interior=average(:mse_interior), mean_bias=average(:mean_bias),
            bias_squared=average(:bias_squared), bias_rmse=sqrt(average(:bias_squared)),
            amplitude_mse=average(:amplitude_mse), amplitude_rmse=sqrt(average(:amplitude_mse)),
            # This remains a signed average of endpoint drifts, not an RMS volume error.
            volume_drift_end=average(:volume_drift_end), training_seconds=training_seconds)
        isempty(intersect(fields, propertynames(statistics))) ||
            throw(ArgumentError("Grouping columns overlap summary statistic names"))
        push!(summaries, (; key..., statistics...))
    end
    isnothing(output_path) || Surge.save_table(output_path, summaries)
    return summaries
end
