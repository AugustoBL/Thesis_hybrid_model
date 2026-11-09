"""
Forecast one event from its initial state without resetting predictions.

Args:
    model: Surrogate mapping features `(space, 5, batch)` to state `(space, 2, batch)`.
    event: Trajectory including initial h [m], u [m/s], and prescribed stress [Pa].
    params: Physical parameters containing face depths [m].

Returns:
    Named tuple with Float32 prediction and target arrays `(N + 1, 2, steps)`, excluding the initial state.
    Channels are cell h [m] with a padded final row, then face u [m/s].
"""
function forecast_event(model, event, params)
    # Each transition consumes the current forcing and targets the next stored state.
    n_steps = length(event) - 1
    first_input = _model_input(first(event), params)
    n_grid_points = size(first_input, 1)
    # Store each predicted state and its reference at the same forecast step.
    prediction = zeros(Float32, n_grid_points, 2, n_steps)
    target = zeros(Float32, n_grid_points, 2, n_steps)
    # Add a singleton batch axis: (space, 2, 1).
    state = reshape(first_input[:, 1:2], n_grid_points, 2, 1)

    for step in 1:n_steps
        model_input = _model_input(event[step], params)
        forcing = reshape(model_input[:, 3:5], n_grid_points, 3, 1)
        # Concatenate predicted h/u and forcing to form (space, 5, 1) input.
        state = model(cat(state, forcing; dims=2))
        # Remove the singleton batch axis for the saved space/channel/time layout.
        prediction[:, :, step] .= state[:, :, 1]
        target[:, :, step] .= _model_target(event[step + 1])
    end
    return (prediction=prediction, target=target)
end

"""
Evaluate saved adapted and pretrained models on held-out events.

Args:
    run: Run record with its directory and test-event membership.
    dataset: Event trajectories, parameters, cell/face grids, and elapsed prediction times [s].
    pretrained_model: Baseline surrogate with matching geometry.
    forecast_hours: Maximum forecast duration [hours].

Returns:
    Per-event metrics, MSE maps `(N + 1, 2, steps)` averaged over test events,
    the first test-event forecast, elapsed times [s], and grids [m].
    Saves evaluation/primary.jld2 and evaluation/metrics.tsv; h MSE is [m^2], u MSE [m^2/s^2].
"""
function evaluate_run(run, dataset, pretrained_model; forecast_hours)
    # Evaluate the validation-selected adapted model saved for this run.
    model = load_model(joinpath(run.directory, "best_model.bson"))
    step_count = searchsortedlast(dataset.times, forecast_hours * 3600)
    event_indices = run.result.split.test
    grid_points = length(dataset.grid.u)
    squared_errors = zeros(Float32, grid_points, 2, step_count)
    baseline_errors = zeros(Float32, grid_points, 2, step_count)
    metrics = NamedTuple[]
    saved_forecast = nothing
    for event_index in event_indices
        event = dataset.solution[event_index][1:step_count+1]
        # Fix LoRA kernels and reuse them throughout each rollout.
        forecast = forecast_event(effective_model(model), event, dataset.params)
        baseline = forecast_event(effective_model(pretrained_model), event, dataset.params)
        # Compare each model against the same numerical reference trajectory.
        error = (forecast.prediction .- forecast.target).^2
        baseline_error = (baseline.prediction .- baseline.target).^2
        # Sum errors across events while retaining spatial, channel, and time axes.
        squared_errors .+= error
        baseline_errors .+= baseline_error
        # Exclude the padded h row from scalar MSE
        water_level_indices = eachindex(dataset.grid.h)
        push!(metrics, (event=event_index,
            mse_h=mean(error[water_level_indices, 1, :]), mse_u=mean(error[:, 2, :]),
            baseline_mse_h=mean(baseline_error[water_level_indices, 1, :]),
            baseline_mse_u=mean(baseline_error[:, 2, :]), finite_rollout=all(isfinite, error)))
        # Save only the first test-event trajectory
        if saved_forecast === nothing
            saved_forecast = (event=event_index, prediction=forecast.prediction,
                target=forecast.target, baseline=baseline.prediction)
        end
    end
    # Average over events only, preserving the (space, 2, steps) map layout.
    result = (metrics=metrics, mse=squared_errors ./ length(event_indices),
        baseline_mse=baseline_errors ./ length(event_indices), forecast=saved_forecast,
        times=dataset.times[1:step_count], grid=dataset.grid)
    output_directory = joinpath(run.directory, "evaluation")
    mkpath(output_directory)
    JLD2.jldsave(joinpath(output_directory, "primary.jld2"); result=result)
    save_table(joinpath(output_directory, "metrics.tsv"), metrics)
    return result
end
