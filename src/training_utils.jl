"""
Count water-level observation indices.

Args:
    stations: Water-level cell indices.

Returns:
    Number of observed water-level cells.
"""
observation_count(stations::AbstractVector) = length(stations)

"""
Count water-level and velocity observation indices.

Args:
    stations: Named tuple of h cell indices and u face indices.

Returns:
    Combined number of h and u observations.
"""
observation_count(stations::NamedTuple) = length(stations.h) + length(stations.u)

"""
Extract water levels at observation stations.

Args:
    state: Full state "(space, 2, batch)".
    stations: Observed water-level cell indices.

Returns:
    Water-level values "(observations, batch)" [m].
"""
extract_station_observations(state, stations::AbstractVector) = state[stations, 1, :]

"""
Extract and stack water-level and velocity observations.

Args:
    state: Full state "(space, 2, batch)".
    stations: Named tuple of h cell indices and u face indices.

Returns:
    Values "(observations, batch)" with h [m] rows followed by u [m/s] rows.
"""
extract_station_observations(state, stations::NamedTuple) =
    vcat(state[stations.h, 1, :], state[stations.u, 2, :])

"""
Convert scalar loss scales to Float32.

Args:
    values: Per-horizon reference error scales.

Returns:
    Float32 vector in the same horizon order.
"""
convert_loss_scales_to_float32(values::AbstractVector{<:Real}) = Float32.(values)

"""
Convert separate water-level and velocity loss scales to Float32.

Args:
    values: Per-horizon named tuples with h, u, and h_count fields.

Returns:
    Vector of h/u scales retaining each h_count partition.
"""
convert_loss_scales_to_float32(values::AbstractVector{<:NamedTuple}) =
    [(h=Float32(v.h), u=Float32(v.u), h_count=v.h_count) for v in values]

"""
Compute normalized endpoint mean-squared error.

Args:
    prediction: Predicted observations "(observations, batch)".
    target: Matching observed targets.
    normalizer: Reference MSE in the target variable"s squared units.

Returns:
    Scalar MSE divided by the reference error scale.
"""
function endpoint_loss(prediction, target, normalizer)
    return sum(abs2, prediction .- target) / (length(target) * normalizer)
end


"""
Compute equally weighted normalized water-level and velocity errors.

Args:
    prediction: Observations "(observations, batch)" with h rows preceding u rows.
    target: Matching h [m] and u [m/s] targets.
    normalizer: Reference h/u MSEs and h_count marking the water-level row count.

Returns:
    Dimensionless mean of the separately normalized h and u MSEs.
"""
function endpoint_loss(prediction, target, normalizer::NamedTuple)
    water_level_station_count = normalizer.h_count
    h = mean(abs2, prediction[1:water_level_station_count, :] .- target[1:water_level_station_count, :]) / normalizer.h
    u = mean(abs2, prediction[water_level_station_count+1:end, :] .- target[water_level_station_count+1:end, :]) / normalizer.u
    # Compare dimensionless variable errors
    return (h + u) / 2
end

"""
Compute fixed pretrained water-level error scales at each horizon.

Args:
    pretrained_model: Base surrogate used to calculate reference errors.
    data: Sparse training-event dataset.
    settings: Configured rollout horizons [steps], window stride, and batch size.
    stations: Observed water-level cell indices.

Returns:
    Reference MSE vector [m^2] in configured-horizon order, floored at 1e-8.
"""
function compute_pretrained_error_scales(pretrained_model, data, settings, stations)
    rollout_states = precompute_rollout_states(pretrained_model, data)
    batches = prepare_window_batches(data, enumerate_training_windows(data, settings), settings, stations)
    squared_errors = zeros(Float64, length(settings.rollout_horizons))
    window_count = 0
    for batch in batches
        set_window_initial_states!(batch, rollout_states)
        state = batch.state
        endpoint_index = 1
        for step in 1:maximum(settings.rollout_horizons)
            state = pretrained_model(cat(state, view(batch.forcing, :, :, step, :); dims=2))
            if step == settings.rollout_horizons[endpoint_index]
                error = state[stations, 1, :] .- view(batch.targets, :, endpoint_index, :)
                squared_errors[endpoint_index] += mean(abs2, error) * length(batch.windows)
                endpoint_index += 1
            end
        end
        window_count += length(batch.windows)
    end
    return max.(squared_errors ./ window_count, 1e-8)
end

"""
Compute fixed pretrained water-level and velocity error scales.

Args:
    pretrained_model: Base surrogate used to calculate reference errors.
    data: Sparse training-event dataset.
    settings: Configured rollout horizons [steps], window stride, and batch size.
    stations: Named tuple of h cell indices and u face indices.

Returns:
    Per-horizon h [m^2], u [m^2/s^2], and h_count tuples; scales are floored at 1e-8.
"""
function compute_pretrained_error_scales(pretrained_model, data, settings, stations::NamedTuple)
    rollout_states = precompute_rollout_states(pretrained_model, data)
    batches = prepare_window_batches(data, enumerate_training_windows(data, settings), settings, stations)
    h_errors = zeros(Float64, length(settings.rollout_horizons))
    u_errors = zeros(Float64, length(settings.rollout_horizons))
    count = 0
    for batch in batches
        set_window_initial_states!(batch, rollout_states)
        state = batch.state
        endpoint = 1
        for step in 1:maximum(settings.rollout_horizons)
            state = pretrained_model(cat(state, view(batch.forcing, :, :, step, :); dims=2))
            if step == settings.rollout_horizons[endpoint]
                error = extract_station_observations(state, stations) .- view(batch.targets, :, endpoint, :)
                water_level_station_count = length(stations.h)
                h_errors[endpoint] += mean(abs2, error[1:water_level_station_count, :]) * length(batch.windows)
                u_errors[endpoint] += mean(abs2, error[water_level_station_count+1:end, :]) * length(batch.windows)
                endpoint += 1
            end
        end
        count += length(batch.windows)
    end
    return [(h=max(h/count, 1e-8), u=max(u/count, 1e-8), h_count=length(stations.h))
            for (h,u) in zip(h_errors,u_errors)]
end

"""
Select the supervised horizons for the current epoch.

Args:
    settings: Rollout horizons, loss mode, fixed endpoint, and curriculum schedule.
    epoch: Current training epoch, starting at one.

Returns:
    Vector of endpoint leads in model steps.
"""
function select_loss_horizons(settings, epoch)
    available_horizons = settings.rollout_horizons
    # If curriculum is enabled, modify the available horizons based on the current epoch
    if settings.curriculum_enabled
        fraction_complete = (epoch - 1) / settings.epochs
        stage = 1
        for boundary in settings.curriculum_boundaries
            if fraction_complete >= boundary
                stage += 1
            end
        end
        available_horizons = available_horizons[1:min(stage, length(available_horizons))]
    end
    # In balanced mode return all available horizons
    if settings.loss_mode == :balanced
        return available_horizons
    end
    # If not in balanced mode but curriculum is enabled return the longest available horizon
    if settings.curriculum_enabled
        return [last(available_horizons)]
    end
    # Else just use the longest horizon
    return [settings.endpoint_horizon]
end

"""
Enumerate training windows with room for the longest rollout.

Args:
    data: Sparse event dataset with forcing "(space, 3, steps, events)".
    settings: Rollout horizons and window stride in model steps.

Returns:
    Vector of "(trajectory_index, start)" pairs in event/window order.
"""
function enumerate_training_windows(data, settings)
    maximum_horizon = maximum(settings.rollout_horizons)
    # Reserve enough future forcing for the longest configured lead
    final_start = size(data.forcing, 3) - maximum_horizon + 1
    return [(trajectory_index, start) for trajectory_index in 1:length(data)
            for start in 1:settings.window_stride:final_start]
end

"""
Prepare forcing and endpoint targets for batches of windows.

Args:
    data: Sparse event dataset.
    windows: Trajectory/start-index pairs.
    settings: Batch size and configured rollout horizons [steps].
    stations: Water-level indices or a named tuple of h/u indices.

Returns:
    Batch records with windows, forcing, targets, and a mutable initial-state buffer.
"""
function prepare_window_batches(data, windows, settings, stations)
    maximum_horizon = maximum(settings.rollout_horizons)
    return map(collect(Iterators.partition(windows, settings.batch_size))) do batch_windows
        grid_points = size(data.initial_input, 1)
        batch_size = length(batch_windows)
        forcing = Array{Float32}(undef, grid_points, 3, maximum_horizon, batch_size)
        targets = Array{Float32}(undef, observation_count(stations), length(settings.rollout_horizons), batch_size)
        state = zeros(Float32, grid_points, 2, batch_size)
        for (batch_index, (trajectory_index, start)) in enumerate(batch_windows)
            forcing[:, :, :, batch_index] .= view(data.forcing, :, :, start:start+maximum_horizon-1, trajectory_index)
            for (horizon_index, horizon) in enumerate(settings.rollout_horizons)

                targets[:, horizon_index, batch_index] .= vec(extract_station_observations(reshape(view(data.target, :, :, start+horizon-1, trajectory_index), grid_points, 2, 1), stations))
            end
        end
        return (windows=collect(batch_windows), forcing=forcing, targets=targets, state=state)
    end
end

"""
Precompute model-generated states for training-window initialization.

Args:
    model: Surrogate advancing h/u from five input features.
    data: Sparse event dataset containing initial states and prescribed forcing.

Returns:
    Float32 states "(space, 2, steps + 1, events)", including the initial state at index one.
"""
function precompute_rollout_states(model, data)
    grid_points, _, trajectory_count = size(data.initial_input)
    step_count = size(data.forcing, 3)
    rollout_states = zeros(Float32, grid_points, 2, step_count + 1, trajectory_count)
    state = copy(data.initial_input[:, 1:2, :])
    # State index 1 is the known initial state; index k+1 holds prediction k.
    rollout_states[:, :, 1, :] .= state
    # Make LoRA kernels once for this rollout.
    model = effective_model(model)
    for step in 1:step_count
        state = model(cat(state, view(data.forcing, :, :, step, :); dims=2))
        rollout_states[:, :, step+1, :] .= state
    end
    return rollout_states
end

"""
Set each training window"s initial state from precomputed predictions.

Args:
    batch: Window batch with trajectory/start-index pairs.
    rollout_states: Precomputed states "(space, 2, steps + 1, events)".
    offset: Additional steps beyond each window origin.

Returns:
    Nothing; modifies batch.state "(space, 2, batch)" in place.
"""
function set_window_initial_states!(batch, rollout_states; offset=0)
    for (batch_index, (event, start)) in enumerate(batch.windows)
        # Copy into the reusable buffer; later objective computation does not
        # mutate the full-event rollout states.
        batch.state[:, :, batch_index] .= view(rollout_states, :, :, start+offset, event)
    end
end


"""
Compute normalized rollout loss at the selected endpoints.

Args:
    model: Surrogate advancing h/u from five input features.
    initial_state: Window initial state "(space, 2, batch)".
    forcing: Prescribed features "(space, 3, steps, batch)".
    targets: Observations "(observations, length(horizons), batch)".
    stations: Water-level indices or a named tuple of h/u indices.
    horizons: Selected forecast leads in model steps.
    normalizers: Scalar or separate h/u reference scales for each selected horizon.
    detached_prefix_steps: Initial transitions evaluated without gradients.

Returns:
    Scalar loss averaged equally across selected horizons.
"""
function rollout_objective(model, initial_state, forcing, targets, stations, horizons, normalizers;
                           detached_prefix_steps=0)

    # Make LoRA kernels once for this rollout.
    model = effective_model(model)
    state = initial_state
    total_loss = zero(eltype(state))
    endpoint_index = 1
    for step in 1:last(horizons)
        # Concatenate predicted h/u with known forcing
        model_input = cat(state, view(forcing, :, :, step, :); dims=2)
        if step <= detached_prefix_steps
            # stop derivatives 
            state = Flux.ignore_derivatives() do
                model(model_input)
            end
        else
            state = model(model_input)
        end
        if endpoint_index <= length(horizons) && step == horizons[endpoint_index]
            # Supervise only selected leads and stations
            total_loss += endpoint_loss(extract_station_observations(state, stations),
                view(targets, :, endpoint_index, :), normalizers[endpoint_index])
            endpoint_index += 1
        end
    end
    return total_loss / length(horizons)
end

"""
Compute sparse water-level MSE over complete event rollouts.

Args:
    model: Surrogate to validate.
    data: Sparse validation-event dataset.
    stations: Observed water-level cell indices.

Returns:
    Water-level MSE [m^2], or Inf for a nonfinite predicted state.
"""
function validation_loss(model, data, stations)
    # Make LoRA kernels once for this rollout
    model = effective_model(model)
    state = copy(data.initial_input[:, 1:2, :])
    squared_error = 0.0
    observation_count = 0
    for step in axes(data.forcing, 3)
        state = model(cat(state, view(data.forcing, :, :, step, :); dims=2))
        # An unstable rollout is an invalid validation
        if !all(isfinite, state)
            return Inf
        end
        if step % data.measurement_gap == 0
            error = state[stations, 1, :] .- data.target[stations, 1, step, :]
            squared_error += sum(abs2, error)
            observation_count += length(error)
        end
    end
    return squared_error / observation_count
end



"""
Prepare window starting states and endpoint targets for one update.

Args:
    batch: Prepared window batch whose state buffer is modified.
    rollout_states: Precomputed model states "(space, 2, steps + 1, events)".
    settings: Configured horizons and gradient-detachment settings.
    horizons: Selected forecast leads [steps].
    normalizers: Reference loss scales in configured-horizon order.

Returns:
    Objective batch with state, forcing, selected targets/horizons/scales, and detached-prefix length.
"""
function prepare_objective_batch!(batch, rollout_states, settings, horizons, normalizers)
    endpoint_indices = [findfirst(==(horizon), settings.rollout_horizons) for horizon in horizons]
    offset = 0
    forcing = batch.forcing
    objective_horizons = horizons
    set_window_initial_states!(batch, rollout_states; offset=offset)
    detached_prefix_steps = 0
    if settings.detach_gradients
        detached_prefix_steps = max(0, last(objective_horizons) - settings.gradient_steps)
    end
    return (state=batch.state, forcing=forcing, targets=batch.targets[:, endpoint_indices, :],
        horizons=objective_horizons, normalizers=convert_loss_scales_to_float32(normalizers[endpoint_indices]),
        detached_prefix_steps=detached_prefix_steps)
end

"""
Differentiate one batch objective and update model parameters.

Args:
    model: Surrogate whose trainable parameters are updated.
    optimizer_state: Flux optimizer state updated in place.
    batch: Prepared objective batch.
    stations: Observed water-level indices or h/u index tuple.
    penalty: Differentiable scalar regularization function of the model.
    timings: Optional dictionary accumulating operation times [s].

Returns:
    Pre-update loss, raw gradient norm, and parameter-update norm.
"""
function optimizer_step!(model, optimizer_state, batch, stations;
                         penalty=model -> 0f0, timings=nothing)
    start_time_ns = time_ns()

    loss, gradients = Flux.withgradient(model) do current_model
        rollout_objective(current_model, batch.state, batch.forcing, batch.targets,
            stations, batch.horizons, batch.normalizers; detached_prefix_steps=batch.detached_prefix_steps) + penalty(current_model)
    end
    isfinite(loss) || error("Nonfinite training loss")
    record_training_time!(timings, :forward_backward, start_time_ns)
    start_time_ns = time_ns()
    # Measure gradients for the history without changing the values passed to Adam.
    gradient_norm = sqrt(sum(sum(abs2, values) for values in Flux.trainables(gradients[1])))
    @assert isfinite(gradient_norm) "Nonfinite gradient"
    previous_parameters = copy_trainable_parameters(model)
    Flux.update!(optimizer_state, model, gradients[1])
    update_norm = sqrt(sum(sum(abs2, parameter .- previous) for (parameter, previous)
        in zip(Flux.trainables(model), previous_parameters)))
    record_training_time!(timings, :optimizer_and_gradient_bookkeeping, start_time_ns)
    return (loss=loss, gradient_norm=gradient_norm, update_norm=update_norm)
end

"""
Accumulate elapsed time for a training phase.

Args:
    timings: Mutable timing dictionary, or nothing to disable recording.
    phase: Dictionary key identifying the measured operation.
    start_time_ns: Starting timestamp from time_ns() [ns].

Returns:
    Nothing; adds elapsed seconds to timings when supplied.
"""
function record_training_time!(timings, phase, start_time_ns)  
    # Convert from nanoseconds to seconds
    isnothing(timings) || (timings[phase] = get(timings, phase, 0.0) + (time_ns() - start_time_ns) / 1e9)
    return nothing
end

"""
Copy all trainable parameter arrays.

Args:
    model: Model whose trainable parameters are copied.

Returns:
    Independent arrays in Flux trainable traversal order.
"""
copy_trainable_parameters(model) = [copy(parameter) for parameter in Flux.trainables(model)]

"""
Restore saved trainable parameter arrays.

Args:
    model: Model receiving the saved values.
    parameters: Saved arrays in matching Flux trainable traversal order.

Returns:
    The model after modifying its trainable arrays in place.
"""
function restore_trainable_parameters!(model, parameters)
    for (parameter, saved_parameter) in zip(Flux.trainables(model), parameters)
        parameter .= saved_parameter
    end
    return model
end

"""
Update the best parameter snapshot on validation improvement.

Args:
    model: Model supplying parameters when the loss improves.
    loss: Current validation loss.
    epoch: Epoch of the current validation score.
    best_parameters: Previously selected parameter arrays.
    best_validation_loss: Lowest validation loss recorded so far.
    best_validation_epoch: Epoch of the previously selected validation score.

Returns:
    Best parameter arrays, validation loss, and validation epoch.
"""
function update_best_checkpoint(model, loss, epoch, best_parameters, best_validation_loss, best_validation_epoch)
    if loss < best_validation_loss
        return copy_trainable_parameters(model), loss, epoch
    end
    return best_parameters, best_validation_loss, best_validation_epoch
end

"""
Check whether validation patience permits stopping.

Args:
    settings: Patience [epochs] and optional minimum_epochs; zero patience disables stopping.
    completed_epoch: Last completed training epoch.
    best_validation_epoch: Epoch with the lowest recorded validation loss.

Returns:
    True when patience and the minimum training budget are satisfied.
"""
function training_should_stop(settings, completed_epoch, best_validation_epoch)
    minimum_epochs = hasproperty(settings, :minimum_epochs) ? settings.minimum_epochs : 0
    return settings.patience > 0 && completed_epoch >= minimum_epochs &&
        completed_epoch - best_validation_epoch >= settings.patience
end

"""
Fit a surrogate using model-generated windows and sparse endpoint losses.

Args:
    model: Surrogate whose trainable parameters are updated.
    training_data: Sparse training-event dataset.
    validation_data: Sparse validation-event dataset.
    settings: Training budget, batching, horizons, normalization, curriculum, gradient, refresh, and stopping settings.
    stations: Training water-level indices or named tuple of h/u indices.
    normalizers: Pretrained reference scales in configured-horizon order.
    optimizer: Flux optimizer specification.
    rng: Random generator for batch order.
    checkpoint: Callback receiving resumable state at validation epochs; defaults to no saving.
    resume: Saved training state, or nothing for a new fit.
    penalty: Differentiable scalar regularization function; defaults to zero.
    timings: Optional dictionary accumulating phase times [s].
    validation_stations: Water-level indices used for validation; defaults to stations.

Returns:
    Resumable model/optimizer/RNG state, history, best parameters, loss/epoch, and update counts.
"""
function train_model!(model, training_data, validation_data; settings, stations, normalizers,
                      optimizer, rng, checkpoint=state -> nothing, resume=nothing,
                      penalty=model -> 0f0, timings=nothing, validation_stations=stations)
    training_start = time_ns()
    prior_seconds = resume === nothing ? 0.0 : resume.training_seconds
    if !settings.normalize_loss
        normalizers = ones(length(settings.rollout_horizons))
    end
    batches = prepare_window_batches(training_data, enumerate_training_windows(training_data, settings), settings, stations)
    if resume === nothing
        optimizer_state = Flux.setup(optimizer, model)
        start_time_ns = time_ns()
        best_validation_loss = validation_loss(model, validation_data, validation_stations)
        record_training_time!(timings, :validation, start_time_ns)
        isfinite(best_validation_loss) || error("Nonfinite initial validation loss")
        best_parameters = copy_trainable_parameters(model)
        best_validation_epoch = 0
        updates = 0
        history = NamedTuple[(epoch=0, updates=0, training_loss=NaN, validation_loss=best_validation_loss,
            gradient_norm=NaN, update_norm=0.0, horizon=0)]
        first_epoch = 1
    else
        # Resume the saved parameter, optimizer, and random-generator state rather than restarting 
        model = resume.model
        optimizer_state = resume.optimizer_state
        rng = resume.rng
        best_parameters = resume.best_parameters
        best_validation_loss = resume.best_loss
        best_validation_epoch = resume.best_epoch
        updates = resume.updates
        history = copy(resume.history)
        first_epoch = resume.epoch + 1
    end
    for epoch in first_epoch:settings.epochs
        if training_should_stop(settings, last(history).epoch, best_validation_epoch)
            break
        end
        start_time_ns = time_ns()
        rollout_states = precompute_rollout_states(model, training_data)
        record_training_time!(timings, :cache, start_time_ns)
        horizons = select_loss_horizons(settings, epoch)
        total_loss = 0.0
        total_gradient_norm = 0.0
        total_update_norm = 0.0
        # Shuffle the order of  batches
        for (update_index, batch_index) in enumerate(randperm(rng, length(batches)))
            if update_index > 1 && (update_index - 1) % settings.cache_refresh_interval == 0
                start_time_ns = time_ns()
                rollout_states = precompute_rollout_states(model, training_data)
                record_training_time!(timings, :cache, start_time_ns)
            end
            batch = prepare_objective_batch!(batches[batch_index], rollout_states, settings, horizons, normalizers)
            step_result = optimizer_step!(model, optimizer_state, batch, stations;
                penalty=penalty, timings=timings)
            updates += 1
            total_loss += step_result.loss
            total_gradient_norm += step_result.gradient_norm
            total_update_norm += step_result.update_norm
        end
        validate = epoch % settings.validation_interval == 0 || epoch == settings.epochs
        current_validation_loss = NaN
        if validate
            start_time_ns = time_ns()
            current_validation_loss = validation_loss(model, validation_data, validation_stations)
            record_training_time!(timings, :validation, start_time_ns)
            isfinite(current_validation_loss) || error("Nonfinite validation loss")
        end
        best_parameters, best_validation_loss, best_validation_epoch = update_best_checkpoint(model,
            current_validation_loss, epoch, best_parameters, best_validation_loss, best_validation_epoch)

        push!(history, (epoch=epoch, updates=updates, training_loss=total_loss / length(batches),
            validation_loss=current_validation_loss, gradient_norm=total_gradient_norm / length(batches),
            update_norm=total_update_norm / length(batches),
            horizon=last(horizons)))
        if validate
            start_time_ns = time_ns()
            checkpoint((model=model, optimizer_state=optimizer_state, rng=rng, history=history,
                best_parameters=best_parameters, best_loss=best_validation_loss, best_epoch=best_validation_epoch,
                updates=updates, epoch=epoch,
                training_seconds=prior_seconds + (time_ns() - training_start) / 1e9))
            record_training_time!(timings, :checkpoint, start_time_ns)
        end
    end
    return (model=model, optimizer_state=optimizer_state, rng=rng, history=history,
        best_parameters=best_parameters, best_loss=best_validation_loss, best_epoch=best_validation_epoch,
        updates=updates, epoch=last(history).epoch,
        training_seconds=prior_seconds + (time_ns() - training_start) / 1e9)
end

function one_step_loss(model, data, batch_size)
    total = 0.0
    for indices in Iterators.partition(axes(data.input, 3), batch_size)
        total += length(indices) * endpoint_loss(model(data.input[:, :, indices]), data.target[:, :, indices], 1)
    end
    return total / size(data.input, 3)
end

function perturb_state_input(input, level, rng)
    level == 0 && return input
    perturbed = copy(input)
    noise_h = level .* randn(rng, Float32, size(input, 1)-1, 1, size(input, 3))
    noise_h .-= mean(noise_h; dims=1) # Do not change the volume the target must conserve.
    perturbed[1:end-1, 1:1, :] .+= noise_h
    perturbed[2:end-1, 2:2, :] .+= level .* randn(rng, Float32, size(input, 1)-2, 1, size(input, 3))
    return perturbed
end

"""Train supervised one-step pairs with the source learning-rate schedule and state noise."""
function train_model!(model, training::NamedTuple, validation::NamedTuple; settings,
                      checkpoint=state -> nothing, resume=nothing)
    if resume === nothing
        optimizer_state = Flux.setup(Adam(settings.learning_rate), model)
        rng = MersenneTwister(settings.seed)
        best_loss = one_step_loss(model, validation, settings.batch_size)
        isfinite(best_loss) || error("Nonfinite initial validation loss")
        best_parameters, best_epoch = copy_trainable_parameters(model), 0
        history = NamedTuple[(epoch=0, learning_rate=settings.learning_rate,
            training_loss=NaN, validation_loss=best_loss)]
        first_epoch = 1
    else
        model, optimizer_state, rng = resume.model, resume.optimizer_state, resume.rng
        history = copy(resume.history)
        best_parameters, best_loss, best_epoch = resume.best_parameters, resume.best_loss, resume.best_epoch
        first_epoch = resume.epoch + 1
    end
    for epoch in first_epoch:settings.epochs
        stage = min(fld(epoch - 1, settings.learning_rate_steps), settings.learning_rate_decay_stages)
        rate = settings.learning_rate * settings.learning_rate_decay^(stage / settings.learning_rate_decay_stages)
        Flux.adjust!(optimizer_state, rate)
        total = 0.0
        for indices in Iterators.partition(randperm(rng, size(training.input, 3)), settings.batch_size)
            input = perturb_state_input(training.input[:, :, indices], settings.noise_std, rng)
            target = training.target[:, :, indices]
            loss, gradient = Flux.withgradient(model) do current
                endpoint_loss(current(input), target, 1)
            end
            isfinite(loss) || error("Nonfinite pretraining loss")
            Flux.update!(optimizer_state, model, gradient[1])
            total += length(indices) * loss
        end
        loss = one_step_loss(model, validation, settings.batch_size)
        isfinite(loss) || error("Nonfinite validation loss")
        best_parameters, best_loss, best_epoch = update_best_checkpoint(model, loss, epoch,
            best_parameters, best_loss, best_epoch)
        push!(history, (epoch=epoch, learning_rate=rate,
            training_loss=total / size(training.input, 3), validation_loss=loss))
        if epoch % settings.checkpoint_interval == 0 || epoch == settings.epochs
            checkpoint((; model, optimizer_state, rng, history, best_parameters, best_loss, best_epoch, epoch))
        end
    end
    return (; model, optimizer_state, rng, history, best_parameters, best_loss, best_epoch, epoch=last(history).epoch)
end
