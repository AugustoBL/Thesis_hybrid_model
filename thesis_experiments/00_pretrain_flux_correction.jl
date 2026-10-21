const repository = normpath(joinpath(@__DIR__, ".."))
const source_file = joinpath(repository, "Data", "24_hour_source.jld2")
const output_directory = joinpath(repository, "results", "experiments", "source_pretraining")
const source_training_events = collect(1:4)
const source_validation_events = collect(29:34)
const source_test_events = collect(35:40)
const hidden_channels = 16
const filter_width = 3
const pretraining_settings = (
    epochs=500, batch_size=50, learning_rate=1e-4,
    learning_rate_decay=0.01, learning_rate_steps=250, learning_rate_decay_stages=3,
    noise_std=0.001f0, checkpoint_interval=25, seed=2026,
)

include(joinpath(@__DIR__, "experiment_dependencies.jl"))

function main()
    source_dataset = Surge.load_burgers_ensemble(basename(source_file);
        data_path=dirname(source_file), relative_times=true)
    time_step = first(source_dataset.times)

    training_pairs = Surge.one_step_pairs(source_dataset, source_training_events)
    validation_pairs = Surge.one_step_pairs(source_dataset, source_validation_events)
    test_pairs = Surge.one_step_pairs(source_dataset, source_test_events)

    specification = (source_file=source_file,
        train=source_training_events, validation=source_validation_events, test=source_test_events,
        settings=pretraining_settings, n_hidden=hidden_channels, n_filter=filter_width,
        dt=time_step, dx=source_dataset.params.dx)
    directory = joinpath(output_directory, "pretraining", "flux_correction")

    Random.seed!(pretraining_settings.seed)
    model = Surge.build_flux_correction_cnn(hidden_channels, filter_width;
        dt=time_step, dx=source_dataset.params.dx)
    record = SourcePretraining.pretrain(model, training_pairs, validation_pairs, directory;
        specification=specification)

    test_loss = Surge.one_step_loss(record.model, test_pairs, pretraining_settings.batch_size)
    JLD2.jldsave(joinpath(directory, "specification.jld2"); specification=specification, test_loss=test_loss)
    summary = [(architecture=:flux_correction, best_epoch=record.result.best_epoch,
        validation_mse=record.result.best_validation_loss, source_test_mse=test_loss,
        trainable_parameters=Surge.trainable_parameter_count(record.model), checkpoint=record.path)]
    Surge.save_table(joinpath(output_directory, "pretraining", "summary.tsv"), summary)
    println("Selected source checkpoint: $(record.path)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
