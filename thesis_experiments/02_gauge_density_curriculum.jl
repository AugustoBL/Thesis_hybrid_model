const repository = normpath(joinpath(@__DIR__, ".."))
const data_directory = joinpath(repository, "Data", "thesis_v4")
const pretrained_file = joinpath(repository, "Trained_models", "flux_correction_pretrained.bson")
const baseline_directory = joinpath(repository, "results", "experiments", "sparse_adaptation")
const output_directory = joinpath(repository, "results", "experiments", "gauge_curriculum")
const reported_cases = [:two_gauges, :curriculum_fast]
const optimizer_settings = (learning_rate=3e-6, betas=(0.9, 0.999), epsilon=1e-8)
const training_settings = (
    epochs=200, batch_size=4,
    rollout_horizons=[16, 32, 64, 128], window_stride=128,
    loss_mode=:balanced, endpoint_horizon=128, normalize_loss=true,
    curriculum_enabled=false, curriculum_boundaries=[0.25, 0.375, 0.5],
    detach_gradients=false, gradient_steps=16,
    validation_interval=5, patience=0, cache_refresh_interval=25,
)

include(joinpath(@__DIR__, "experiment_dependencies.jl"))
include(joinpath(@__DIR__, "shared", "gauge_curriculum_analysis.jl"))
import .GaugeCurriculumAnalysis

function configuration(case, temporal)
    stations = [2, 4, 97, 99]
    settings = training_settings
    if case == :two_gauges
        stations = [2, 99]
        if temporal == :long
            settings = merge(settings, (loss_mode=:endpoint,))
        end
    elseif case == :curriculum_fast
        settings = merge(settings, (curriculum_enabled=true,
            curriculum_boundaries=[0.05, 0.10, 0.15]))
    end
    return (
        experiment=6, name="gauge_curriculum", case=case,
        output_directory=output_directory,
        dataset_file=joinpath(data_directory, "uniform.jld2"),
        pretrained_file=pretrained_file, measurement_gap=16, training_steps=1024,
        stations=stations, observation_noise_std=0.0,
        training_noise_seed_offset=710000, validation_noise_seed_offset=720000,
        optimizer_seed_offset=10000, optimizer=optimizer_settings,
        adaptation=(method=:full, ranks=nothing), training=settings,
    )
end

function main()
    dataset = Surge.load_burgers_ensemble("uniform.jld2"; data_path=data_directory, relative_times=true)
    pretrained_model = Surge.load_model(pretrained_file)
    event_splits = Surge.make_event_splits(18;
        fold_count=5, folds=1:5, repeats=1:3,
        training_event_count=2, validation_event_count=4,
        selection_seed=42000, training_seed=420000)
    selected_objectives = Dict(fold => first(Surge.load_table(joinpath(baseline_directory,
        "selections", "temporal_fold_$fold.tsv"))).candidate for fold in 1:5)

    jobs = NamedTuple[]
    index_rows = NamedTuple[]
    for case in reported_cases, event_split in event_splits
        settings = configuration(case, selected_objectives[event_split.fold])
        directory = joinpath(output_directory, "runs", string(case),
            "fold_$(event_split.fold)", "repeat_$(event_split.repeat)")
        result = Surge.fit_run(pretrained_model, dataset, event_split, settings, directory)
        push!(jobs, (case=case, split=event_split, stations=settings.stations, directory=directory))
        push!(index_rows, (case=case, fold=event_split.fold, repeat=event_split.repeat,
            best_epoch=result.best_epoch, training_seconds=result.training_seconds,
            updates=result.updates, directory=directory))
    end
    Surge.save_table(joinpath(output_directory, "indices", "reported_cases.tsv"), index_rows)

    metrics = NamedTuple[]
    for job in jobs
        path = joinpath(job.directory, "evaluation", "interval_metrics.jld2")
        if isfile(path)
            rows = JLD2.load(path, "rows")
        else
            models = [:adapted => Surge.load_model(joinpath(job.directory, "best_model.bson")),
                :pretrained => pretrained_model]
            rows = ExperimentDiagnostics.evaluate_models(models, dataset, job.split, job.stations;
                include_finite=true).summaries
            mkpath(dirname(path))
            JLD2.jldsave(path * ".tmp"; rows)
            mv(path * ".tmp", path; force=true)
            Surge.save_table(joinpath(dirname(path), "interval_metrics.tsv"), rows)
        end
        for row in rows
            push!(metrics, (case=job.case, fold=job.split.fold, repeat=job.split.repeat, row...))
        end
    end

    # Read paired baseline scores from the main campaign; no duplicate forecasts or reference files.
    for event_split in event_splits
        temporal = selected_objectives[event_split.fold]
        for (case, objective, layout) in ((:four_gauges, temporal, :both),
                (:eight_gauges, temporal, :eight), (:balanced, :balanced, :both),
                (:curriculum, :curriculum, :both))
            name = "full__$(objective)__uniform__$(layout)__noise0cm"
            path = joinpath(baseline_directory, "runs", name,
                "fold_$(event_split.fold)", "repeat_$(event_split.repeat)", "diagnostics", "metrics.tsv")
            for row in Surge.load_table(path)
                identity = (model=row.model, event=row.event, finite_rollout=all(isfinite, values(row)[3:end]))
                scores = Base.structdiff(row, (model=row.model, event=row.event))
                push!(metrics, (case=case, fold=event_split.fold, repeat=event_split.repeat, identity..., scores...))
            end
        end
    end

    event_means, summaries = GaugeCurriculumAnalysis.aggregate(metrics)
    Surge.save_table(joinpath(output_directory, "event_metrics.tsv"), metrics)
    Surge.save_table(joinpath(output_directory, "event_means.tsv"), event_means)
    Surge.save_table(joinpath(output_directory, "summary.tsv"), summaries)
    Surge.save_table(joinpath(output_directory, "comparisons.tsv"), GaugeCurriculumAnalysis.comparisons(event_means))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
