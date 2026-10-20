"""
Roll out one event from its initial state.

Args:
    model: Surrogate mapping `(space, 5, batch)` input to h/u state.
    event: Trajectory including initial h [m], u [m/s], and prescribed stress [Pa].
    params: Physical parameters containing face depths [m].

Returns:
    Tuple of prediction and target arrays `(space, 2, steps)`, excluding the initial state.
"""
function _rollout_event(model, event, params)
    # Each transition consumes the current forcing and targets the next stored state.
    n_steps = length(event) - 1
    first_input = _model_input(first(event), params)
    n_grid_points = size(first_input, 1)
    prediction = zeros(Float32, n_grid_points, 2, n_steps)
    target = zeros(Float32, n_grid_points, 2, n_steps)
    # Introduce a batch of one while retaining both physical state channels.
    state = reshape(first_input[:, 1:2], n_grid_points, 2, 1)

    for step in 1:n_steps
        model_input = _model_input(event[step], params)
        # Use only prescribed features from the reference event at subsequent times;
        # its h/u fields never enter the autoregressive model input.
        forcing = reshape(model_input[:, 3:5], n_grid_points, 3, 1)
        # Feed the previous prediction back as state, advancing one native model step.
        state = model(cat(state, forcing; dims=2))
        # Remove the singleton batch axis for the saved space/channel/time layout.
        prediction[:, :, step] .= state[:, :, 1]
        target[:, :, step] .= _model_target(event[step + 1])
    end
    return prediction, target
end

"""
Forecast one event without resetting the predicted state.

Args:
    model: Surrogate mapping `(space, 5, batch)` input to h/u state.
    event: Trajectory including initial h [m], u [m/s], and prescribed stress [Pa].
    params: Physical parameters containing face depths [m].

Returns:
    Named tuple with prediction and target arrays `(space, 2, steps)`.
"""
function forecast_event(model, event, params)
    prediction, target = _rollout_event(model, event, params)
    return (prediction=prediction, target=target)
end

"""
Evaluate saved adapted and pretrained models on held-out events.

Args:
    run: Run record with its directory and test-event membership.
    dataset: Event trajectories, parameters, grids, and prediction times [s].
    pretrained_model: Baseline surrogate with matching geometry.
    forecast_hours: Maximum forecast duration [hours].

Returns:
    Metrics, adapted/baseline MSE maps, one example forecast, times, and grids; saves evaluation files.
"""
function evaluate_run(run, dataset, pretrained_model; forecast_hours)
    model = load_model(joinpath(run.directory, "best_model.bson"))
    # Select all forecast output times up to the requested cutoff. The
    # dataset times already omit the initial state.
    step_count = searchsortedlast(dataset.times, forecast_hours * 3600)
    event_indices = run.result.split.test
    grid_points = length(dataset.grid.u)
    squared_errors = zeros(Float32, grid_points, 2, step_count)
    baseline_errors = zeros(Float32, grid_points, 2, step_count)
    metrics = NamedTuple[]
    saved_forecast = nothing
    for event_index in event_indices
        # Keep one extra trajectory record because the initial condition precedes
        # the first forecast output.
        event = dataset.solution[event_index][1:step_count+1]
        forecast = forecast_event(effective_model(model), event, dataset.params)
        baseline = forecast_event(effective_model(pretrained_model), event, dataset.params)
        error = (forecast.prediction .- forecast.target).^2
        baseline_error = (baseline.prediction .- baseline.target).^2
        # Accumulate dense pointwise error first; averaging across events later
        # preserves both spatial and lead-time variation.
        squared_errors .+= error
        baseline_errors .+= baseline_error
        # Water level lives on cells, so exclude its dummy final face-grid row
        # from physical MSE. Velocity uses every face, including walls.
        water_level_indices = eachindex(dataset.grid.h)
        push!(metrics, (event=event_index,
            mse_h=mean(error[water_level_indices, 1, :]), mse_u=mean(error[:, 2, :]),
            baseline_mse_h=mean(baseline_error[water_level_indices, 1, :]),
            baseline_mse_u=mean(baseline_error[:, 2, :]), finite_rollout=all(isfinite, error)))
        # Keep one representative trajectory to limit output size; aggregate
        # metrics still use all test events.
        if saved_forecast === nothing
            saved_forecast = (event=event_index, prediction=forecast.prediction,
                target=forecast.target, baseline=baseline.prediction)
        end
    end
    # Normalize each map by the number of events rather than averaging away
    # its spatial or temporal axes.
    result = (metrics=metrics, mse=squared_errors ./ length(event_indices),
        baseline_mse=baseline_errors ./ length(event_indices), forecast=saved_forecast,
        times=dataset.times[1:step_count], grid=dataset.grid)
    output_directory = joinpath(run.directory, "evaluation")
    mkpath(output_directory)
    JLD2.jldsave(joinpath(output_directory, "primary.jld2"); result=result)
    save_table(joinpath(output_directory, "metrics.tsv"), metrics)
    return result
end
