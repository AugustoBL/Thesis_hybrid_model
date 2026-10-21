"""
Write named records as a tab-separated table.

Args:
    path: Destination filepath; existing output is overwritten.
    rows: Nonempty collection of records sharing named columns.

Returns:
    Output filepath.
"""
function save_table(path, rows; delimiter='\t')
    mkpath(dirname(path))
    open(path, "w") do output
        columns = propertynames(first(rows))
        println(output, join(string.(columns), delimiter))
        for row in rows
            println(output, join((getproperty(row, column) for column in columns), delimiter))
        end
    end
    return path
end

"""Read a numeric/symbol TSV written by save_table."""
function load_table(path)
    lines = readlines(path)
    columns = Tuple(Symbol.(split(first(lines), '\t')))
    parse_field = value -> value == "true" ? true : value == "false" ? false :
        something(tryparse(Int, value), tryparse(Float64, value), Symbol(value))
    return [NamedTuple{columns}(Tuple(parse_field.(split(line, '\t')))) for line in lines[2:end]]
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
    @assert all(length(dataset.solution[index]) > configuration.training_steps
        for index in vcat(split.train, split.validation))
    training = prepare_sparse_station_event_data(dataset.solution[split.train], dataset.params;
        measurement_gap=configuration.measurement_gap,
        observed_grid_indices_by_channel=(configuration.stations, Int[]),
        max_steps=configuration.training_steps)
    validation = prepare_sparse_station_event_data(dataset.solution[split.validation], dataset.params;
        measurement_gap=configuration.measurement_gap,
        observed_grid_indices_by_channel=(configuration.stations, Int[]),
        max_steps=configuration.training_steps)
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
function fit_run(pretrained_model, dataset, split, configuration, directory; save_final=false)
    result_path = joinpath(directory, "result.jld2")
    checkpoint_path = joinpath(directory, "checkpoint.jld2")
    if isfile(result_path)
        result = JLD2.load(result_path, "result")
        isequal(result.configuration, configuration) && isequal(result.split, split) ||
            error("Saved run settings differ: $directory")
        return result
    end
    mkpath(directory)
    previous = isfile(checkpoint_path) ? JLD2.load(checkpoint_path, "saved") : nothing
    if previous !== nothing
        isequal(previous.configuration, configuration) && isequal(previous.split, split) ||
            error("Checkpoint settings differ: $directory")
    end
    preparation_start = time_ns()
    training_data, validation_data = prepare_training_split(dataset, split, configuration)
    normalizers = compute_pretrained_error_scales(pretrained_model, training_data,
        configuration.training, configuration.stations)
    ranks = configuration.adaptation.method == :lora ? configuration.adaptation.ranks[split.fold] : nothing
    model = configure_adaptation(pretrained_model, configuration.adaptation.method; seed=split.seed, ranks=ranks)
    preparation_seconds = (time_ns() - preparation_start) / 1e9 +
        (previous === nothing ? 0.0 : previous.preparation_seconds)
    save_checkpoint = function(state)
        saved = (; state, configuration, split, preparation_seconds)
        JLD2.jldsave(checkpoint_path * ".tmp"; saved)
        mv(checkpoint_path * ".tmp", checkpoint_path; force=true)
        save_table(joinpath(directory, "training_history.tsv"), state.history)
        println("  epoch $(state.epoch)/$(configuration.training.epochs), validation=$(state.best_loss)")
        flush(stdout)
    end
    state = train_model!(model, training_data, validation_data;
        settings=configuration.training, stations=configuration.stations, normalizers=normalizers,
        optimizer=Adam(configuration.optimizer.learning_rate,
            configuration.optimizer.betas, configuration.optimizer.epsilon),
        rng=MersenneTwister(split.seed + configuration.optimizer_seed_offset),
        resume=previous === nothing ? nothing : previous.state, checkpoint=save_checkpoint)
    save_final && save_model(joinpath(directory, "final_model.bson"), state.model)
    restore_trainable_parameters!(state.model, state.best_parameters)
    save_model(joinpath(directory, "best_model.bson"), state.model)
    result = (configuration=configuration, split=split, history=state.history,
        best_validation_loss=state.best_loss, best_epoch=state.best_epoch,
        completed_epochs=state.epoch, updates=state.updates,
        trainable_parameters=trainable_parameter_count(state.model), normalizers=normalizers,
        training_seconds=state.training_seconds, preparation_seconds=preparation_seconds)
    JLD2.jldsave(result_path * ".tmp"; result)
    mv(result_path * ".tmp", result_path; force=true)
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
function run_experiment(pretrained_model, dataset, splits; configuration, save_final=false)
    runs = NamedTuple[]
    for split in splits
        directory = joinpath(configuration.output_directory, configuration.name,
            "experiment_$(configuration.experiment)", string(configuration.adaptation.method),
            "fold_$(split.fold)", "repeat_$(split.repeat)")
        result = fit_run(pretrained_model, dataset, split, configuration, directory; save_final)
        push!(runs, (directory=directory, result=result))
    end
    return runs
end
