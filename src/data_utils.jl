"""
Locate the repository data directory.

Args:
    None.

Returns:
    Path to the Data directory.
"""
get_data_path() = joinpath(@__DIR__, "..", "Data")
"""
Locate the repository model directory.

Args:
    None.

Returns:
    Path to the Trained_models directory.
"""
get_model_path() = joinpath(@__DIR__, "..", "Trained_models")

"""
Load a saved ensemble and omit the initial timestamp.

Args:
    filename: Ensemble filename.
    data_path: Directory containing the ensemble.

Returns:
    Named tuple containing solution, grid, prediction times in seconds, and params.
"""
function load_burgers_ensemble(filename::String; data_path=get_data_path(), relative_times=false)
    data = load(joinpath(data_path, filename))
    return (
        solution=data["solution"],
        grid=data["grid"],
        times=relative_times ? Float64.(data["times"][2:end]) .- Float64(first(data["times"])) : data["times"][2:end],
        params=data["params"],
    )
end

"""
Encode one surge state as model input.

Args:
    state: State with N water levels [m], N + 1 velocities [m/s], and wind stress [Pa].
    params: Physical parameters containing face depths D [m].

Returns:
    Float32 array `(N + 1, 5)` containing padded h, u, depth/30, stress, and boundary indicators.
"""
function _model_input(state, params)
    n_water_level_points = length(state.h)
    n_velocity_points = length(state.u)
    input = zeros(Float32, n_velocity_points, 5)
    input[1:n_water_level_points, 1] .= state.h
    input[:, 2] .= state.u
    # Depth normalization
    input[:, 3] .= params.D ./ 30f0
    input[:, 4] .= state.tau
    input[1, 5] = 1f0
    input[end, 5] = 1f0
    return input
end

"""
Encode water level and velocity as a model target.

Args:
    state: State containing water levels h [m] and face velocities u [m/s].

Returns:
    Float32 array `(N + 1, 2)` with a padded final water-level row.
"""
function _model_target(state)
    n_water_level_points = length(state.h)
    n_velocity_points = length(state.u)
    target = zeros(Float32, n_velocity_points, 2)
    target[1:n_water_level_points, 1] .= state.h
    target[:, 2] .= state.u
    return target
end

"""
Store initial states, forcing, and sparse observation targets.

Args:
    initial_input: Initial features `(space, 5, events)`.
    forcing: Depth/30, stress, and boundary features `(space, 3, steps, events)`.
    target: Observed h/u values `(space, 2, steps, events)`; unobserved entries are placeholders.
    observation_mask: Array matching target, with ones at observed entries.
    measurement_gap: Number of model steps between observations.

Returns:
    A sparse event dataset retaining the supplied arrays.
"""
struct SparseStationEventDataset{TI, TF, TT, TM}
    initial_input::TI
    forcing::TF
    target::TT
    observation_mask::TM
    measurement_gap::Int
end

"""
Count the events in a sparse dataset.

Args:
    dataset: Sparse event dataset.

Returns:
    Number of events.
"""
Base.length(dataset::SparseStationEventDataset) = size(dataset.initial_input, 3)

"""
Resolve observation indices for each state channel.

Args:
    n_channels: Number of state channels.
    shared_grid_indices: Optional indices shared by the selected channels.
    channels: Channels using shared indices.
    grid_indices_by_channel: Optional per-channel indices overriding the shared specification.

Returns:
    Per-channel collections of spatial indices.
"""
function _resolve_grid_indices_by_channel(
    n_channels::Int;
    shared_grid_indices::Union{Nothing, AbstractVector{<:Integer}}=nothing,
    channels::AbstractVector{<:Integer}=collect(1:n_channels),
    grid_indices_by_channel=nothing,
)
    if grid_indices_by_channel === nothing
        return [
            channel in channels && shared_grid_indices !== nothing ?
                shared_grid_indices : Int[]
            for channel in 1:n_channels
        ]
    end
    return grid_indices_by_channel
end

"""
Pack event trajectories into forcing and sparse observations.

Args:
    events: Trajectories including their initial states.
    params: Physical parameters containing face depths D [m].
    measurement_gap: Observation spacing in model steps.
    observed_grid_indices: Optional spatial indices shared across observed channels.
    observed_channels: Channels using shared indices: 1 for h, 2 for u.
    observed_grid_indices_by_channel: Optional per-channel indices overriding shared indices.
    max_steps: Optional maximum number of transitions to retain.

Returns:
    SparseStationEventDataset with Float32 arrays and complete initial states.
"""
function prepare_sparse_station_event_data(
    events,
    params;
    measurement_gap::Int,
    observed_grid_indices::Union{Nothing, AbstractVector{<:Integer}}=nothing,
    observed_channels::AbstractVector{<:Integer}=[1, 2],
    observed_grid_indices_by_channel=nothing,
    max_steps::Union{Nothing, Integer}=nothing,
)
    n_events = length(events)
    # A trajectory with T+1 stored states supplies T one-step transitions.
    available_steps = length(first(events)) - 1
    n_steps = max_steps === nothing ? available_steps : min(Int(max_steps), available_steps)
    first_state = first(first(events))
    first_input = _model_input(first_state, params)
    n_grid_points = size(first_input, 1)
    observation_grid_indices = _resolve_grid_indices_by_channel(
        2;
        shared_grid_indices=observed_grid_indices,
        channels=observed_channels,
        grid_indices_by_channel=observed_grid_indices_by_channel,
    )

    initial_input = zeros(Float32, n_grid_points, 5, n_events)
    # Separate known inputs from state targets; wind may change every step
    # even though observations arrive only at the measurement interval.
    forcing = zeros(Float32, n_grid_points, 3, n_steps, n_events)
    target = zeros(Float32, n_grid_points, 2, n_steps, n_events)
    observation_mask = zeros(Float32, size(target))

    for event_index in eachindex(events)
        event = events[event_index]
        initial_input[:, :, event_index] .=
            _model_input(first(event), params)

        for step in 1:n_steps
            model_input = _model_input(event[step], params)
            forcing[:, :, step, event_index] .= model_input[:, 3:5]
            # Retain forcing for every transition but populate supervision only
            # at observation times; omitted targets and their masks remain zero.
            step % measurement_gap == 0 || continue

            # The kth forcing slice advances to the (k+1)th stored state.
            full_target = _model_target(event[step + 1])
            for channel in axes(target, 2)
                channel_grid_indices = observation_grid_indices[channel]
                isempty(channel_grid_indices) && continue
                target[channel_grid_indices, channel, step, event_index] .=
                    full_target[channel_grid_indices, channel]
                # Record availability independently of the value, since a true observation
                # can itself equal zero.
                observation_mask[channel_grid_indices, channel, step, event_index] .= 1f0
            end
        end
    end

    return SparseStationEventDataset(
        initial_input,
        forcing,
        target,
        observation_mask,
        measurement_gap,
    )
end

"""
Add Gaussian noise to observed targets.

Args:
    dataset: Sparse event dataset.
    noise_std_by_channel: Standard deviations per channel: metres for h and m/s for u.
    rng: Random generator for observation noise.

Returns:
    Dataset with copied noisy targets and unchanged initial states, forcing, and mask.
"""
function add_observation_noise(
    dataset::SparseStationEventDataset,
    noise_std_by_channel;
    rng::AbstractRNG=Random.default_rng(),
)
    noisy_target = copy(dataset.target)
    for channel in axes(noisy_target, 2)
        noise_std = Float32(noise_std_by_channel[channel])
        noise_std == 0 && continue
        channel_mask = view(dataset.observation_mask, :, channel, :, :)
        channel_target = view(noisy_target, :, channel, :, :)
        # Draw in the channel shape and mask afterwards; unobserved placeholders
        # remain unchanged and each observed time/location/event gets its own noise.
        channel_target .+= noise_std .* randn(rng, Float32, size(channel_target)) .* channel_mask
    end
    return SparseStationEventDataset(
        dataset.initial_input,
        dataset.forcing,
        noisy_target,
        dataset.observation_mask,
        dataset.measurement_gap,
    )
end

"""
Build reproducible event-level training, validation, and test splits.

Args:
    event_count: Total number of events.
    fold_count: Number of disjoint test folds.
    folds: Fold identifiers to include.
    repeats: Training-draw repeat identifiers.
    training_event_count: Number of training events per repeat.
    validation_event_count: Number of validation events per fold.
    selection_seed: Seed for training-event draws.
    training_seed: Base seed for model initialization and optimization.

Returns:
    Named tuples containing fold, repeat, event memberships, training pool, and seed.
"""
function make_event_splits(event_count; fold_count, folds, repeats, training_event_count,
                           validation_event_count, selection_seed, training_seed)
    # Integer partition boundaries allocate all events even when the event
    # count is not divisible by the number of folds.
    test_blocks = [collect(fld((fold - 1) * event_count, fold_count) + 1:
        fld(fold * event_count, fold_count)) for fold in 1:fold_count]
    splits = NamedTuple[]
    for fold in folds
        remaining_events = vcat([test_blocks[mod1(fold + offset, fold_count)]
            for offset in 1:fold_count-1]...)
        # Keep validation deterministic within a fold; only the training subset
        # changes between repeats.
        validation_events = remaining_events[1:validation_event_count]
        # Exclude both held-out groups before sampling to prevent event leakage.
        training_pool = sort(setdiff(1:event_count, vcat(test_blocks[fold], validation_events)))
        for repeat in repeats
            random_generator = MersenneTwister(selection_seed + 100 * fold + repeat)
            training_events = sort(shuffle(random_generator, training_pool)[1:training_event_count])
            push!(splits, (fold=fold, repeat=repeat, train=training_events,
                validation=validation_events, test=test_blocks[fold], pool=training_pool,
                seed=training_seed + 100 * fold + repeat))
        end
    end
    return splits
end

"""Encode consecutive event states as supervised one-step input/target arrays."""
function one_step_pairs(dataset, indices)
    count = sum(length(dataset.solution[i]) - 1 for i in indices)
    faces = length(dataset.grid.u)
    inputs = Array{Float32}(undef, faces, 5, count)
    targets = Array{Float32}(undef, faces, 2, count)
    sample = 0
    for event_index in indices
        event = dataset.solution[event_index]
        for step in 1:length(event)-1
            sample += 1
            inputs[:, :, sample] .= _model_input(event[step], dataset.params)
            targets[:, :, sample] .= _model_target(event[step+1])
        end
    end
    return (input=inputs, target=targets)
end
