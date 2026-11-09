module GaugeCurriculumAnalysis

using ..ExperimentDiagnostics, Statistics

function aggregate(rows)
    # Average repeats for each case, model, test event and interval.
    event_columns = (:case, :model, :event, :from_hours, :to_hours, :repeats,
        :finite_rollout, :mse_h, :mse_u, :anomaly_mse, :bias_squared)
    events = [NamedTuple{event_columns}(Tuple(getproperty(row, key) for key in event_columns))
        for row in ExperimentDiagnostics.average_event_errors(rows;
            group_by=(:case, :model, :event, :from_hours, :to_hours))]
    sort!(events; by=row -> (string(row.case), string(row.model), row.event, row.from_hours))
    # Give events equal weight and take RMSE after averaging squared errors.
    summaries = [(case=row.case, model=row.model, from_hours=row.from_hours, to_hours=row.to_hours,
        events=row.events, minimum_repeats=row.repeats_min, finite_rollout=row.finite_rollout,
        rmse_h=row.rmse_h, rmse_u=row.rmse_u, anomaly_rmse=sqrt(row.anomaly_mse), bias_rmse=row.bias_rmse)
        for row in ExperimentDiagnostics.summarize_run_diagnostics(rows;
            group_by=(:case, :model, :from_hours, :to_hours))]
    sort!(summaries; by=row -> (string(row.case), string(row.model), row.from_hours))
    return events, summaries
end

"""Compare paired events and check that test events match."""
function comparisons(events)
    # Compare two gauges and the faster curriculum with the main experiment controls.
    pairs = [(:two_gauges, :four_gauges), (:two_gauges, :eight_gauges),
        (:curriculum_fast, :balanced), (:curriculum_fast, :curriculum)]
    rows = NamedTuple[]
    intervals = sort(unique((row.from_hours, row.to_hours) for row in events))
    for (case, reference) in pairs, (lower, upper) in intervals
        # Compare adapted models over the same interval.
        selected = filter(row -> row.case == case && row.model == :adapted &&
            row.from_hours == lower && row.to_hours == upper, events)
        baseline = filter(row -> row.case == reference && row.model == :adapted &&
            row.from_hours == lower && row.to_hours == upper, events)
        sort!(selected; by=row -> row.event)
        sort!(baseline; by=row -> row.event)
        # Check that test events and repeat counts match before paired comparisons.
        [row.event for row in selected] == [row.event for row in baseline] ||
            error("Unpaired test events in $case versus $reference")
        [row.repeats for row in selected] == [row.repeats for row in baseline] ||
            error("Unpaired repeats in $case versus $reference")
        # Take RMSE after averaging events. A negative change means lower error.
        rmse = sqrt(mean(row.mse_h for row in selected))
        reference_rmse = sqrt(mean(row.mse_h for row in baseline))
        push!(rows, (case=case, model=:adapted, reference=reference, reference_model=:adapted,
            from_hours=lower, to_hours=upper, events=length(selected),
            rmse_h=rmse, reference_rmse_h=reference_rmse, change_percent=100 * (rmse / reference_rmse - 1),
            events_improved=count(a.mse_h < b.mse_h for (a, b) in zip(selected, baseline)),
            finite_rollout=all(row.finite_rollout for row in vcat(selected, baseline))))
    end
    return rows
end

end
