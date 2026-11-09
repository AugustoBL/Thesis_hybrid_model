function endpoint_loss(prediction, target, normalizer)
    return sum(abs2, prediction .- target) / (length(target) * normalizer)
end


"""
Compute equally weighted normalized water-level and velocity errors.

Args:
    prediction: Observations "(observations, batch)" with h rows preceding u rows.
    target: Matching h [m] and u [m/s] targets.
    normalizer: Reference h/u MSEs and h_count giving the water-level entry count.

Returns:
    Dimensionless mean of the separately normalized h and u MSEs.
"""
function endpoint_loss(prediction, target, normalizer::NamedTuple)
    water_level_station_count = normalizer.h_count
    h = mean(abs2, prediction[1:water_level_station_count, :] .- target[1:water_level_station_count, :]) / normalizer.h
    u = mean(abs2, prediction[water_level_station_count+1:end, :] .- target[water_level_station_count+1:end, :]) / normalizer.u
    return (h + u) / 2
end

"""
Compute pretrained water-level error scales at each horizon.

Args:
    pretrained_model: Pretrained model used to calculate reference errors.
    data: Training data set.
    settings: Rollout horizons [steps], window stride, and batch size.
    stations: Observed water-level cell indices.

Returns:
    Reference MSE vector [m^2] in configured-horizon order, with floor at 1e-8.
"""
function compute_pretrained_error_scales(pretrained_model, data, settings, stations)
    rollout_states = precompute_rollout_states(pretrained_model, data)

    batches = prepare_window_batches(data, enumerate_training_windows(data, settings), settings, stations)
    isempty(batches) && throw(ArgumentError("No training windows fit the configured rollout horizons"))

    squared_errors = zeros(Float64, length(settings.rollout_horizons))
    window_count = 0
    for batch in batches
        set_window_initial_states!(batch, rollout_states)

        state = batch.state
        endpoint_index = 1

        for step in 1:maximum(settings.rollout_horizons)
            state = pretrained_model(cat(state, view(batch.forcing, :, :, step, :); dims=2))
            # If it is the endstep of a horizon
            if step == settings.rollout_horizons[endpoint_index]
                error = state[stations, 1, :] .- view(batch.targets, :, endpoint_index, :)

                squared_errors[endpoint_index] += mean(abs2, error) * length(batch.windows)
                endpoint_index += 1
            end
        end
        window_count += length(batch.windows)
    end
    return max.(squared_errors ./ window_count, 1e-8) # gives every window equal weight even if the batch sizes do not match.
end

"""
Compute pretrained water-level and velocity error scales.

Args:
    pretrained_model: Pretrained model used to calculate reference errors.
    data: Training dataset.
    settings: Rollout horizons [steps], window stride, and batch size.
    stations: h cell indices and u face indices.

Returns:
    Per-horizon h [m^2], u [m^2/s^2], and h_count; scales are floored at 1e-8.
"""
function compute_pretrained_error_scales(pretrained_model, data, settings, stations::NamedTuple)
    rollout_states = precompute_rollout_states(pretrained_model, data)

    batches = prepare_window_batches(data, enumerate_training_windows(data, settings), settings, stations)
    isempty(batches) && throw(ArgumentError("No training windows fit the configured rollout horizons"))

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
Select the training horizons for the current epoch.

Args:
    settings: Rollout horizons, loss mode, fixed endpoint, and curriculum schedule.
    epoch: Current training epoch

Returns:
    Vector of endpoint leads in model steps.
"""
function select_loss_horizons(settings, epoch)
    available_horizons = settings.rollout_horizons
    # If curriculum is enabled, modify the available horizons based on the current epoch
    if settings.curriculum_enabled
        fraction_complete = (epoch - 1) / settings.epochs # fraction of completed epochs
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
    # Without curriculum, use the configured fixed endpoint.
    return [settings.endpoint_horizon]
end

"""
Enumerate training windows.

Args:
    data: Sparse event dataset containing forcing "(space, 3, steps, events)".
    settings: Rollout horizons and window stride in model steps.

Returns:
    Vector of "(trajectory_index, start)" pairs in window order.
"""
function enumerate_training_windows(data, settings)
    maximum_horizon = maximum(settings.rollout_horizons)
    # Reserve enough space for the longest lead
    final_start = size(data.forcing, 3) - maximum_horizon + 1
    # Store (trajectory index, starting step) pairs
    windows = Tuple{Int, Int}[]

    for trajectory_index in 1:length(data)
        for start in 1:settings.window_stride:final_start
            window = (trajectory_index, start)
            push!(windows, window)
        end
    end
    return windows
end

"""
Prepare forcing and endpoint targets for batches of windows.

Args:
    data: Sparse event dataset.
    windows: Trajectory/start index pairs.
    settings: Batch size and configured rollout horizons
    stations: Water-level indices or a named tuple of h/u indices.

Returns:
    Records containing windows, forcing `(space, 3, max_horizon, batch)`,
    targets `(observations, horizons, batch)`, and zero state `(space, 2, batch)`.
"""
function prepare_window_batches(data, windows, settings, stations)
    maximum_horizon = maximum(settings.rollout_horizons)
    grid_points = size(data.initial_input, 1)
    batches = []

     # Select the windows in this batch.
    for first_index in 1:settings.batch_size:length(windows)
        # find last index to prevent it from going beyond the available windows.
        last_index = min(first_index + settings.batch_size - 1, length(windows))
        # Selected windows
        batch_windows = windows[first_index:last_index]
        batch_size = length(batch_windows)

        # Allocate arrays
        forcing = Array{Float32}(
            undef, grid_points, 3, maximum_horizon, batch_size
        )
        targets = Array{Float32}(
            undef, observation_count(stations),
            length(settings.rollout_horizons), batch_size
        )
        state = zeros(Float32, grid_points, 2, batch_size)

        for (batch_index, (trajectory_index, start)) in enumerate(batch_windows)
            # Copy forcing for this window (all points, all features, timesteps for this window, one trajectory)
            forcing[:, :, :, batch_index] .= view(
                data.forcing, :, :,
                start:start + maximum_horizon - 1,
                trajectory_index
            )

            # Copy observations at each requested horizon
            for (horizon_index, horizon) in enumerate(settings.rollout_horizons)
                # The timestep where we have a target
                target_step = start + horizon - 1
                # Copy target (all points, all features, one timestep, one trajectory)
                target_state = view(
                    data.target, :, :, target_step, trajectory_index
                )
                # reshape (gridpoints, variables) to (gridpoints, variables,batch)
                target_state = reshape(target_state, grid_points, 2, 1)
                # Extract only the values on where there are observations
                observations = extract_station_observations(target_state, stations)

                targets[:, horizon_index, batch_index] .= vec(observations)
            end
        end
        batch = (
            windows=batch_windows,
            forcing=forcing,
            targets=targets,
            state=state
        )
        push!(batches, batch)
    end
    return batches
end

"""
Precompute model states for training-window initialization.

Args:
    model: Model
    data: Sparse event dataset containing initial states and prescribed forcing.

Returns:
    Float32 states `(space, 2, steps + 1, events)`; index 1 is the initial state.
"""
function precompute_rollout_states(model, data)
    grid_points, _, trajectory_count = size(data.initial_input)
    step_count = size(data.forcing, 3)
    rollout_states = zeros(Float32, grid_points, 2, step_count + 1, trajectory_count)
    state = copy(data.initial_input[:, 1:2, :])
    # State index k + 1 holds the prediction after step k.
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
Fill each training window's initial state from a precomputed rollout.

Args:
    batch: Window batch
    rollout_states: Precomputed states "(space, 2, steps + 1, events)".

Returns:
    Nothing; modifies batch.state "(space, 2, batch)" in place.
"""

function set_window_initial_states!(batch, rollout_states)
    for (batch_index, (event, start)) in enumerate(batch.windows)
        batch.state[:, :, batch_index] .=
            view(rollout_states, :, :, start, event)
    end
end

"""
Compute normalized rollout loss at the selected endpoints.

Args:
    model: Model
    initial_state: Initial state "(space, 2, batch)".
    forcing: Forcing "(space, 3, steps, batch)".
    targets: Observations "(observations, length(horizons), batch)".
    stations: Observation locations
    horizons: Nonempty, strictly increasing forecast leads [steps].
    normalizers: Reference scales for each selected horizon.
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
            state = Flux.ignore_derivatives() do
                model(model_input)
            end
        else
            state = model(model_input)
        end
        if endpoint_index <= length(horizons) && step == horizons[endpoint_index]
            # Calc loss when reaching a selected lead endpoint
            total_loss += endpoint_loss(extract_station_observations(state, stations),
                view(targets, :, endpoint_index, :), normalizers[endpoint_index])
            endpoint_index += 1
        end
    end
    return total_loss / length(horizons)
end

"""
Compute water-level MSE at measurement steps over the full validation trajectory.

Args:
    model: Surrogate to validate.
    data: Sparse validation-event dataset.
    stations: Observed water-level cell indices.

Returns:
    Water-level MSE [m^2], or Inf for an unstable rollout.
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

"""Validate water levels using the h indices from a mixed h/u station specification."""
validation_loss(model, data, stations::NamedTuple) = validation_loss(model, data, stations.h)


"""
Prepare window starting states and endpoint targets for one update.

Args:
    batch: Window batch whose initial-state buffer is filled in place.
    rollout_states: Precomputed model states "(space, 2, steps + 1, events)".
    settings: Configured horizons and gradient-detachment settings.
    horizons: Selected forecast leads [steps].
    normalizers: Reference loss scales in configured-horizon order.

Returns:
    Objective batch with state, forcing, selected targets/horizons/scales, and detached-prefix length.
"""
function prepare_objective_batch!(batch, rollout_states, settings, horizons, normalizers)
    endpoint_indices = [findfirst(==(horizon), settings.rollout_horizons) for horizon in horizons]
    forcing = batch.forcing
    set_window_initial_states!(batch, rollout_states)
    detached_prefix_steps = 0
    if settings.detach_gradients
        detached_prefix_steps = max(0, last(horizons) - settings.gradient_steps)
    end
    return (state=batch.state, forcing=forcing, targets=batch.targets[:, endpoint_indices, :],
        horizons=horizons, normalizers=convert_loss_scales_to_float32(normalizers[endpoint_indices]),
        detached_prefix_steps=detached_prefix_steps)
end

"""
Update model parameters.

Args:
    model: Model
    optimizer_state: Flux optimizer state updated in place.
    batch: Prepared objective batch.
    stations: Observed water-level indices or h/u
    timings: Optional dictionary for the operation times [s].

Returns:
    Named tuple containing the pre-update loss.
"""
function optimizer_step!(model, optimizer_state, batch, stations;
                         timings=nothing)
    start_time_ns = time_ns()

    # Compute loss and gradients
    loss, gradients = Flux.withgradient(model) do current_model
        rollout_objective(current_model, batch.state, batch.forcing, batch.targets,
            stations, batch.horizons, batch.normalizers; detached_prefix_steps=batch.detached_prefix_steps)
    end
    isfinite(loss) || error("Nonfinite training loss")
    # Record propagation time
    record_training_time!(timings, :forward_backward, start_time_ns)
    start_time_ns = time_ns()
    # Stop training if there are nonfinite gradients
    @assert all(g -> all(isfinite, g), Flux.trainables(gradients[1])) "Nonfinite gradient"
    Flux.update!(optimizer_state, model, gradients[1])
    record_training_time!(timings, :optimizer_and_gradient_bookkeeping, start_time_ns)
    return (loss=loss,)
end

"""
Accumulate elapsed time for a training phase.

Args:
    timings: Mutable timing dictionary, or nothing to disable recording.
    phase: Dictionary key identifying the measured operation.
    start_time_ns: Starting timestamp from time_ns() [ns].

Returns:
    Nothing.
"""
function record_training_time!(timings, phase, start_time_ns)
    # Convert from nanoseconds to seconds
    isnothing(timings) || (timings[phase] = get(timings, phase, 0.0) + (time_ns() - start_time_ns) / 1e9)
    return nothing
end

"""
Copy all trainable parameters.

Args:
    model: Model

Returns:
    A copy of the trainable parameters.
"""
copy_trainable_parameters(model) = [copy(parameter) for parameter in Flux.trainables(model)]

"""
Restore saved trainable parameter arrays.

Args:
    model: Model
    parameters: Saved arrays in Flux.trainables(model) order, with matching shapes.

Returns:
    The model after modifying its trainable arrays in place.
"""
function restore_trainable_parameters!(model, parameters)
    trainable_parameters = Flux.trainables(model)
    length(trainable_parameters) == length(parameters) || throw(DimensionMismatch("Parameter count differs"))
    all(size(parameter) == size(saved) for (parameter, saved) in zip(trainable_parameters, parameters)) ||
        throw(DimensionMismatch("Parameter shapes differ"))
    for (parameter, saved_parameter) in zip(trainable_parameters, parameters)
        parameter .= saved_parameter
    end
    return model
end

"""
Update the best parameter snapshot on validation improvement.

Args:
    model: Model
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
Check whether early-stopping patience has been reached.

Args:
    settings: Patience and minimum_epochs (optional). Zero patience disables early stopping.
    completed_epoch: Last completed training epoch.
    best_validation_epoch: Epoch with the lowest recorded validation loss.

Returns:
    True when patience and the minimum training budget are satisfied.
"""
function training_should_stop(settings, completed_epoch, best_validation_epoch)
    # Honor an optional minimum training budget.
    minimum_epochs = hasproperty(settings, :minimum_epochs) ? settings.minimum_epochs : 0
    # A patience of zero or less disables early stopping
    if settings.patience <= 0
        return false
    end
    # Allow the minimum number of epochs to finish
    if completed_epoch < minimum_epochs
        return false
    end
    # Count epochs since the best validation result
    epochs_without_improvement = completed_epoch - best_validation_epoch

    if epochs_without_improvement >= settings.patience
        return true
    end
    return false
end

"""
Train with model-initialized rollout windows and endpoint losses.

Args:
    model: Model
    training_data: Sparse training-event dataset.
    validation_data: Sparse validation-event dataset.
    settings: Training budget, batching, horizons, normalization, curriculum, gradient, and stopping settings.
    stations: Training water-level indices or named tuple of h/u indices.
    normalizers: Pretrained reference scales.
    optimizer: Flux optimizer specification.
    rng: Random generator for batch order.
    checkpoint: Callback receiving training state at validation epochs; defaults to no saving.
    timings: Optional dictionary accumulating phase times [s].
    validation_stations: Water-level indices or h/u station record (uses h); defaults to stations.

Returns:
    Current model/optimizer/RNG state, history, best parameters, loss/epoch, and update counts.
"""
function train_model!(model, training_data, validation_data; settings, stations, normalizers,
                      optimizer, rng, checkpoint=state -> nothing, timings=nothing, validation_stations=stations)

    training_start = time_ns()
    # A unit scale disables reference-error normalization.
    if !settings.normalize_loss
        normalizers = ones(length(settings.rollout_horizons))
    end
    batches = prepare_window_batches(training_data, enumerate_training_windows(training_data, settings), settings, stations)
    isempty(batches) && throw(ArgumentError("No training windows fit the configured rollout horizons"))

    optimizer_state = Flux.setup(optimizer, model)
    start_time_ns = time_ns()
    best_validation_loss = validation_loss(model, validation_data, validation_stations)
    record_training_time!(timings, :validation, start_time_ns)
    isfinite(best_validation_loss) || error("Nonfinite initial validation loss")
    best_parameters = copy_trainable_parameters(model)
    best_validation_epoch = 0
    updates = 0
    history = NamedTuple[(epoch=0, updates=0, training_loss=NaN, validation_loss=best_validation_loss,
        horizon=0)]
    # Stop before a new epoch when patience is exhausted.
    for epoch in 1:settings.epochs
        if training_should_stop(settings, last(history).epoch, best_validation_epoch)
            break
        end
        start_time_ns = time_ns()
        # Compute the initial states for each training rollout
        rollout_states = precompute_rollout_states(model, training_data)
        record_training_time!(timings, :cache, start_time_ns)

        horizons = select_loss_horizons(settings, epoch)
        total_loss = 0.0
        # Shuffle the order of  batches
        for batch_index in randperm(rng, length(batches))
            batch = prepare_objective_batch!(batches[batch_index], rollout_states, settings, horizons, normalizers)

            step_result = optimizer_step!(model, optimizer_state, batch, stations;
                timings=timings)
            updates += 1
            total_loss += step_result.loss
        end
        # Validate when we reach a validation timestep
        validate = epoch % settings.validation_interval == 0 || epoch == settings.epochs
        current_validation_loss = NaN
        if validate
            start_time_ns = time_ns()
            current_validation_loss = validation_loss(model, validation_data, validation_stations)
            record_training_time!(timings, :validation, start_time_ns)
            isfinite(current_validation_loss) || error("Nonfinite validation loss")
        end
        # Snapshot parameters when validation improves.
        best_parameters, best_validation_loss, best_validation_epoch = update_best_checkpoint(model,
            current_validation_loss, epoch, best_parameters, best_validation_loss, best_validation_epoch)

        push!(history, (epoch=epoch, updates=updates, training_loss=total_loss / length(batches),
            validation_loss=current_validation_loss,
            horizon=last(horizons)))
        if validate
            start_time_ns = time_ns()
            # Save training progress, including the best parameters found so far
            checkpoint((model=model, optimizer_state=optimizer_state, rng=rng, history=history,
                best_parameters=best_parameters, best_loss=best_validation_loss, best_epoch=best_validation_epoch,
                updates=updates, epoch=epoch,
                training_seconds = (time_ns() - training_start) / 1e9))
            record_training_time!(timings, :checkpoint, start_time_ns)
        end
    end
    return (model=model, optimizer_state=optimizer_state, rng=rng, history=history,
        best_parameters=best_parameters, best_loss=best_validation_loss, best_epoch=best_validation_epoch,
        updates=updates, epoch=last(history).epoch,
        training_seconds = (time_ns() - training_start) / 1e9)
end

"""
Compute average one-step prediction loss over batches.

Args:
    model: Model predicting the next state from an input.
    data: Input and target arrays with samples along the third dimension.
    batch_size: Maximum number of samples per batch.

Returns:
    Mean endpoint loss across all samples, weighted by actual batch size.
"""
function one_step_loss(model, data, batch_size)
    total = 0.0
    sample_count = size(data.input, 3)

    for first_index in 1:batch_size:sample_count
        # Find the last sample in this batch
        last_index = min(first_index + batch_size - 1, sample_count)
        indices = first_index:last_index
        current_batch_size = length(indices)

        # Select inputs and corresponding targets
        inputs = data.input[:, :, indices]
        targets = data.target[:, :, indices]

        # Predict one step and calculate the batch loss
        predictions = model(inputs)
        batch_loss = endpoint_loss(predictions, targets, 1)

        # Weight by the actual number of samples
        total += current_batch_size * batch_loss
    end
    # Average over all samples
    average_loss = total / sample_count
    return average_loss
end

"""
Add random noise to water levels and interior velocities.

Args:
    input: Input array with dimensions (space, features, batch).
    level: Noise amplitude. Zero returns the original input unchanged.
    rng: Random generator used to sample Gaussian noise.

Returns:
    Copy with zero-mean spatial water-level noise and unchanged boundary velocities.
    The final water-level row is padding and remains unchanged.
"""
function perturb_state_input(input, level, rng)
    level == 0 && return input
    perturbed = copy(input)
    noise_h = level .* randn(rng, Float32, size(input, 1)-1, 1, size(input, 3))
    noise_h .-= mean(noise_h; dims=1) # Do not change the volume the target must conserve.
    perturbed[1:end-1, 1:1, :] .+= noise_h
    perturbed[2:end-1, 2:2, :] .+= level .* randn(rng, Float32, size(input, 1)-2, 1, size(input, 3))
    return perturbed
end

"""
Train a model using noisy one-step inputs and clean targets.

Args:
    model: Model predicting state `(space, 2, batch)` from input features.
    training: Input `(space, features, samples)` and target `(space, 2, samples)` arrays.
    validation: Clean input/target arrays with the same layout as training.
    settings: Epochs, batching, learning-rate schedule, noise, seed, and checkpoint interval.
    checkpoint: Callback receiving training state; defaults to no saving.

Returns:
    Current model/optimizer/RNG state, history, and best parameter snapshot, loss, and epoch.
"""
function train_model!(model, training::NamedTuple, validation::NamedTuple; settings,
                      checkpoint=state -> nothing)

    optimizer_state = Flux.setup(Adam(settings.learning_rate), model)
    rng = MersenneTwister(settings.seed)
    best_loss = one_step_loss(model, validation, settings.batch_size)
    isfinite(best_loss) || error("Nonfinite initial validation loss")
    best_parameters, best_epoch = copy_trainable_parameters(model), 0
    history = NamedTuple[(epoch=0, learning_rate=settings.learning_rate,
        training_loss=NaN, validation_loss=best_loss)]


    for epoch in 1:settings.epochs
        completed_epochs = epoch - 1

        # Count how many learning-rate intervals have passed
        stage = fld(completed_epochs, settings.learning_rate_steps)

        # Limit the stage to the configured maximum
        if stage > settings.learning_rate_decay_stages
            stage = settings.learning_rate_decay_stages
        end
        rate = settings.learning_rate_decay_stages == 0 ? settings.learning_rate :
            settings.learning_rate * settings.learning_rate_decay^(stage / settings.learning_rate_decay_stages)
        # Optimisers requires the adjusted rate to retain its original number type.
        rate = convert(typeof(settings.learning_rate), rate)
        Flux.adjust!(optimizer_state, rate)

        total = 0.0
        # Shuffle the sample indices once per epoch
        sample_count = size(training.input, 3)
        shuffled_indices = randperm(rng, sample_count)

        for first_index in 1:settings.batch_size:sample_count
            # Select this batch from the shuffled indices
            last_index = min(first_index + settings.batch_size - 1, sample_count)
            indices = shuffled_indices[first_index:last_index]
            current_batch_size = length(indices)

            # Prepare noisy inputs and corresponding targets
            input = perturb_state_input(
                training.input[:, :, indices], settings.noise_std, rng
            )
            target = training.target[:, :, indices]

            # Calculate the loss and gradients
            loss, gradient = Flux.withgradient(model) do current
                predictions = current(input)
                endpoint_loss(predictions, target, 1)
            end
            if !isfinite(loss)
                error("Nonfinite pretraining loss")
            end
            # Update the model parameters
            Flux.update!(optimizer_state, model, gradient[1])

            # Accumulate loss weighted by the actual batch size
            total += current_batch_size * loss
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
    stations: h cell indices and u face indices.

Returns:
    Values "(observations, batch)" with h [m] rows followed by u [m/s] rows.
"""
extract_station_observations(state, stations::NamedTuple) =
    vcat(state[stations.h, 1, :], state[stations.u, 2, :])

"""
Convert a vector of scalar reference MSEs to Float32.

Args:
    values: Reference MSE vector in horizon order.

Returns:
    Float32 reference MSE vector.
"""
convert_loss_scales_to_float32(values::AbstractVector{<:Real}) = Float32.(values)

"""
Convert separate water-level and velocity loss scales to Float32.

Args:
    values: Vector of (h, u, h_count) records in horizon order.

Returns:
    Records with Float32 h/u scales and unchanged h_count.
"""
convert_loss_scales_to_float32(values::AbstractVector{<:NamedTuple}) =
    [(h=Float32(v.h), u=Float32(v.u), h_count=v.h_count) for v in values]

"""
Compute normalized endpoint mean-squared error.

Args:
    prediction: Predicted observations "(observations, batch)".
    target: Observed targets with the same shape as prediction.
    normalizer: Reference MSE; use 1 for unnormalized MSE.

Returns:
    Scalar MSE divided by the reference error scale.
"""
