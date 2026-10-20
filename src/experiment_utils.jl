"""
Write named records as a tab-separated table.

Args:
    path: Destination filepath; existing output is overwritten.
    rows: Nonempty collection of records sharing named columns.

Returns:
    Output filepath.
"""
function save_table(path, rows)
    mkpath(dirname(path))
    open(path, "w") do output
        # Use a consistent named-tuple field order so summaries remain readable
        # without needing a separate schema file.
        columns = propertynames(first(rows))
        println(output, join(string.(columns), '\t'))
        for row in rows
            println(output, join((getproperty(row, column) for column in columns), '\t'))
        end
    end
    return path
end

"""
Prepare sparse water-level training and validation observations.

Args:
    dataset: Full event ensemble and physical parameters.
    split: Training/validation event indices and random seed.
    configuration: Training duration [steps], stations, measurement gap, and noise settings [m].

Returns:
    Training and validation datasets with noise-free initial states and configured observation noise.
"""
function prepare_training_split(dataset, split, configuration)
    # Check both supervised splits before building arrays; test events are
    # reserved for later evaluation and are not prepared here.
    event_indices = vcat(split.train, split.validation)
    @assert all(length(dataset.solution[index]) > configuration.training_steps for index in event_indices)
    training = prepare_sparse_station_event_data(dataset.solution[split.train], dataset.params;
        measurement_gap=configuration.measurement_gap,
        # Explicitly supervise only h; the empty second collection leaves every
        # velocity target unobserved, although velocity is still evolved.
        observed_grid_indices_by_channel=(configuration.stations, Int[]),
        max_steps=configuration.training_steps)
    validation = prepare_sparse_station_event_data(dataset.solution[split.validation], dataset.params;
        measurement_gap=configuration.measurement_gap,
        # Explicitly supervise only h; the empty second collection leaves every
        # velocity target unobserved, although velocity is still evolved.
        observed_grid_indices_by_channel=(configuration.stations, Int[]),
        max_steps=configuration.training_steps)
    # Independent deterministic seed offsets prevent reusing the same noise
    # realization for training and validation.
    training = add_observation_noise(training, (configuration.observation_noise_std, 0.0);
        rng=MersenneTwister(split.seed + configuration.training_noise_seed_offset))
    validation = add_observation_noise(validation, (configuration.observation_noise_std, 0.0);
        rng=MersenneTwister(split.seed + configuration.validation_noise_seed_offset))
    return training, validation
end

"""
Load, resume, or train one adaptation run and save its models.

Args:
    pretrained_model: Base surrogate for adaptation and reference error scales.
    dataset: Full event ensemble.
    split: Event memberships, fold, repeat, and seed.
    configuration: Adaptation, optimizer, observation, and training settings.
    directory: Directory for checkpoints, models, and run metadata.

Returns:
    Run result containing configuration, split, history, validation selection, counts, and loss scales.
"""
function fit_run(pretrained_model, dataset, split, configuration, directory)
    completion_path = joinpath(directory, "result.jld2")
    # A result record is the completion marker; this shortcut trusts the
    # existing run directory rather than rechecking all saved model artifacts.
    if isfile(completion_path)
        return JLD2.load(completion_path, "result")
    end
    mkpath(directory)
    training_data, validation_data = prepare_training_split(dataset, split, configuration)
    ranks = nothing
    if configuration.adaptation.method == :lora
        # Use ranks selected for this fold, keeping held-out fold selection separate.
        ranks = configuration.adaptation.ranks[split.fold]
    end
    model = configure_adaptation(pretrained_model, configuration.adaptation.method;
        seed=split.seed, ranks=ranks)
    # Set fixed objective scales from pretrained training-window errors,
    # not from validation or test outcomes.
    normalizers = compute_pretrained_error_scales(pretrained_model, training_data,
        configuration.training, configuration.stations)
    optimizer = Adam(configuration.optimizer.learning_rate,
        configuration.optimizer.betas, configuration.optimizer.epsilon)
    checkpoint_path = joinpath(directory, "checkpoint.jld2")
    resume = nothing
    if isfile(checkpoint_path)
        # Restore optimizer moments and RNG as well as parameters via train_model!.
        resume = JLD2.load(checkpoint_path, "state")
    end
    # The trainer invokes this at validation epochs, coupling resumable state
    # and the human-readable learning curve to the same training point.
    """
    Save resumable training state and its history table.

    Args:
        state: Model, optimizer, random generator, history, and validation selection.

    Returns:
        Path to the saved training-history table.
    """
    save_checkpoint = function(state)
        JLD2.jldsave(checkpoint_path; state=state)
        save_table(joinpath(directory, "training_history.tsv"), state.history)
    end
    state = train_model!(model, training_data, validation_data;
        settings=configuration.training, stations=configuration.stations,
        normalizers=normalizers, optimizer=optimizer,
        rng=MersenneTwister(split.seed + configuration.optimizer_seed_offset),
        resume=resume, checkpoint=save_checkpoint)
    # Preserve the actual final iterate before loading the validation-best snapshot.
    save_model(joinpath(directory, "final_model.bson"), state.model)
    # Only trainable arrays changed during fitting; frozen base arrays already
    # match the selected model and need no separate restoration.
    restore_trainable_parameters!(state.model, state.best_parameters)
    save_model(joinpath(directory, "best_model.bson"), state.model)
    result = (configuration=configuration, split=split, history=state.history,
        best_validation_loss=state.best_loss, best_epoch=state.best_epoch,
        completed_epochs=state.epoch, updates=state.updates,
        trainable_parameters=trainable_parameter_count(state.model), normalizers=normalizers)
    # Write completion metadata after both model exports so subsequent calls
    # can distinguish a finished fit from an in-progress checkpoint.
    JLD2.jldsave(completion_path; result=result)
    return result
end

"""
Fit the configured adaptation for each event split.

Args:
    pretrained_model: Base surrogate to adapt.
    dataset: Full event ensemble and physical parameters.
    splits: Fold/repeat training, validation, and test memberships.
    configuration: Input paths, output root, experiment identity, and fitting settings.

Returns:
    Vector of directory/result records; completed runs are reused.
"""
function run_experiment(pretrained_model, dataset, splits; configuration)
    dataset_checksum = sha256(read(configuration.dataset_file))
    model_checksum = sha256(read(configuration.pretrained_file))
    runs = NamedTuple[]
    for split in splits
        # Include event membership and input contents in the identifier so runs
        # with different data/model bytes do not silently share a result directory.
        configuration_id = bytes2hex(sha256(repr((configuration, split, dataset_checksum, model_checksum))))[1:12]
        directory = joinpath(configuration.output_directory, configuration.name,
            "experiment_$(configuration.experiment)", string(configuration.adaptation.method),
            configuration_id, "fold_$(split.fold)", "repeat_$(split.repeat)")
        result = fit_run(pretrained_model, dataset, split, configuration, directory)
        push!(runs, (directory=directory, result=result))
    end
    return runs
end
