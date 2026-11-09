const repository = normpath(joinpath(@__DIR__, ".."))
const data_directory = joinpath(repository, "Data", "thesis_v4")
const pretrained_file = joinpath(repository, "Trained_models", "flux_correction_pretrained.bson")
const output_directory = joinpath(repository, "results", "experiments", "source_domain_control")
const station_indices = [2, 4, 97, 99]
const optimizer_settings = (learning_rate=3e-6, betas=(0.9, 0.999), epsilon=1e-8)
const training_settings = (
    epochs=200, batch_size=4,
    rollout_horizons=[16, 32, 64, 128], window_stride=128,
    loss_mode=:endpoint, endpoint_horizon=128, normalize_loss=true,
    curriculum_enabled=false, curriculum_boundaries=[0.25, 0.375, 0.5],
    detach_gradients=false, gradient_steps=1,
    validation_interval=5, patience=0, cache_refresh_interval=25,
)
const configuration = (
    experiment=5, name="source_temporal_controls", output_directory=output_directory,
    dataset_file=joinpath(data_directory, "control.jld2"), pretrained_file=pretrained_file,
    measurement_gap=16, training_steps=1024, stations=station_indices,
    observation_noise_std=0.0, training_noise_seed_offset=710000,
    validation_noise_seed_offset=720000, optimizer_seed_offset=10000,
    optimizer=optimizer_settings, adaptation=(method=:full, ranks=nothing),
    training=training_settings,
)

include(joinpath(@__DIR__, "experiment_dependencies.jl"))
include(joinpath(@__DIR__, "src", "source_control_analysis.jl"))

function main()
    # Experiment 5: adapt without hidden drag and compare with the pretrained model.
    pretrained_model = Surge.load_model(pretrained_file)
    source_dataset = Surge.load_surge_dataset(basename(configuration.dataset_file);
        data_path=dirname(configuration.dataset_file))
    target_dataset = Surge.load_surge_dataset(basename(joinpath(data_directory, "uniform.jld2"));
        data_path=dirname(joinpath(data_directory, "uniform.jld2")))
    # Use the same events and repeat seeds as the main experiments.
    event_splits = Surge.make_event_splits(18;
        fold_count=5, folds=collect(1:5), repeats=collect(1:3),
        training_event_count=2, validation_event_count=4,
        selection_seed=42000, training_seed=420000)

    # Train full adaptation at lead 128 on each source split.
    jobs = NamedTuple[]
    for event_split in event_splits
        directory = joinpath(output_directory, "runs", "long128",
            "fold_$(event_split.fold)", "repeat_$(event_split.repeat)")
        result = Surge.fit_run(pretrained_model, source_dataset, event_split, configuration, directory)
        push!(jobs, (objective=:long128, fold=event_split.fold, repeat=event_split.repeat,
            directory=directory, best_epoch=result.best_epoch,
            best_validation_loss=result.best_validation_loss,
            training_seconds=result.training_seconds, preparation_seconds=result.preparation_seconds))
        Surge.save_table(joinpath(output_directory, "run_index.tsv"), jobs)
    end

    # Evaluate source test events or load their saved interval scores.
    metrics = NamedTuple[]
    for job in jobs
        path = joinpath(job.directory, "diagnostics", "metrics.tsv")
        if isfile(path)
            rows = Surge.load_table(path)
        else
            report = ExperimentDiagnostics.diagnose_saved_run(job.directory, source_dataset;
                pretrained_model=pretrained_model, forecast_hours=72.0,
                interior_indices=collect(9:92), interval_boundaries=[0.0, 12.0, 24.0, 48.0, 72.0],
                save_series=false)
            rows = report.summaries
        end
        for row in rows
            push!(metrics, (objective=job.objective, fold=job.fold, repeat=job.repeat, row...))
        end
    end
    Surge.save_table(joinpath(output_directory, "event_metrics.tsv"), metrics)
    # Average repeats for each source test event and compare with the pretrained model.
    events, summaries = SourceControlAnalysis.summarize_source_control(metrics, jobs)
    Surge.save_table(joinpath(output_directory, "event_means.tsv"), events)
    Surge.save_table(joinpath(output_directory, "summary.tsv"), summaries)

    # Test source-adapted models on uniform hidden drag without training again.
    target_output = joinpath(repository, "results", "experiments", "source_to_target")
    target_rows = SourceControlAnalysis.evaluate_transfer(jobs, target_dataset, pretrained_model, target_output;
        forecast_hours=72.0, boundaries=[0.0, 12.0, 24.0, 48.0, 72.0],
        stations=collect(1:100), interior_indices=collect(9:92))
    summary = SourceControlAnalysis.summarize_transfer(target_rows)
    Surge.save_table(joinpath(target_output, "summary.csv"), summary; delimiter=',')

end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
