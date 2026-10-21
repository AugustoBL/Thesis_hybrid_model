const repository = normpath(joinpath(@__DIR__, ".."))
const data_directory = joinpath(repository, "Data", "thesis_v4")
const pretrained_file = joinpath(repository, "Trained_models", "flux_correction_pretrained.bson")
const output_directory = joinpath(repository, "results", "experiments", "sparse_adaptation")

const adaptation_methods = [:full, :output, :last_two, :first_two, :lora, :film]
const selection_objectives = [:short, :long, :balanced, :curriculum, :detached]
const additional_objectives = [:fixed32, :fixed64, :one_step]
const station_layouts = Dict(
    :both => [2, 4, 97, 99],
    :one => [2, 4, 6, 8],
    :eight => [2, 4, 6, 8, 93, 95, 97, 99],
    :spread => [10, 37, 64, 91],
)

const geometry_cases = [
    (:uniform, :both), (:uniform, :one), (:uniform, :eight), (:uniform, :spread),
    (:interior, :both), (:interior, :one), (:interior, :spread), (:boundary, :both),
]
const noise_standard_deviations = [0.01, 0.05, 0.10, 0.20]
const optimizer_settings = (learning_rate=3e-6, betas=(0.9, 0.999), epsilon=1e-8)
const training_settings = (
    epochs=200,
    batch_size=4,
    rollout_horizons=[16, 32, 64, 128],
    window_stride=128,
    loss_mode=:balanced,
    endpoint_horizon=128,
    normalize_loss=true,
    curriculum_enabled=false,
    curriculum_boundaries=[0.25, 0.375, 0.5],
    detach_gradients=false,
    gradient_steps=16,
    validation_interval=5,
    patience=0,
    cache_refresh_interval=25,
)

include(joinpath(@__DIR__, "experiment_dependencies.jl"))
include(joinpath(repository, "lora_rank_optimisation", "rank_selection.jl"))

function objective_settings(objective)
    if objective == :balanced
        return training_settings
    elseif objective == :curriculum
        return merge(training_settings, (curriculum_enabled=true,))
    end

    endpoint = 128
    if objective == :short
        endpoint = 16
    elseif objective == :fixed32
        endpoint = 32
    elseif objective == :fixed64
        endpoint = 64
    end
    detach_gradients = objective == :detached || objective == :one_step
    gradient_steps = 16
    if objective == :one_step
        gradient_steps = 1
    end

    return merge(training_settings, (
        loss_mode=:endpoint,
        endpoint_horizon=endpoint,
        detach_gradients=detach_gradients,
        gradient_steps=gradient_steps,
    ))
end

function configuration(method, objective, profile, layout, noise_std, ranks)
    return (
        experiment=0, name="flux_correction",
        output_directory=output_directory,
        dataset_file=joinpath(data_directory, "$profile.jld2"),
        pretrained_file=pretrained_file,
        measurement_gap=16, training_steps=1024,
        stations=station_layouts[layout], observation_noise_std=noise_std,
        training_noise_seed_offset=710000, validation_noise_seed_offset=720000,
        optimizer_seed_offset=10000, optimizer=optimizer_settings,
        adaptation=(method=method, ranks=ranks), training=objective_settings(objective),
    )
end

function run_condition!(jobs, pretrained_model, datasets, ranks,
        stage, method, objective, profile, layout, noise_std, split)
    noise_cm = round(Int, noise_std * 100)
    name = "$(method)__$(objective)__$(profile)__$(layout)__noise$(noise_cm)cm"
    directory = joinpath(output_directory, "runs", name,
        "fold_$(split.fold)", "repeat_$(split.repeat)")
    settings = configuration(method, objective, profile, layout, noise_std, ranks)
    println("Stage $stage: $name, fold $(split.fold), repeat $(split.repeat)")
    result = Surge.fit_run(pretrained_model, datasets[profile], split, settings, directory)
    push!(jobs, (stage=stage, method=method, temporal=objective, profile=profile,
        layout=layout, noise_std=noise_std, fold=split.fold, repeat=split.repeat,
        directory=directory, best_validation_loss=result.best_validation_loss,
        best_epoch=result.best_epoch,
        training_seconds=result.training_seconds, preparation_seconds=result.preparation_seconds))
    Surge.save_table(joinpath(output_directory, "run_index.tsv"), jobs)
    return result.best_validation_loss
end

function main()
    pretrained_model = Surge.load_model(pretrained_file)
    datasets = Dict{Symbol, Any}()
    for profile in (:uniform, :interior, :boundary)
        dataset = Surge.load_burgers_ensemble(basename(joinpath(data_directory, "$profile.jld2"));
            data_path=dirname(joinpath(data_directory, "$profile.jld2")), relative_times=true)
        datasets[profile] = dataset
    end
    event_splits = Surge.make_event_splits(18;
        fold_count=5, folds=collect(1:5), repeats=collect(1:3),
        training_event_count=2, validation_event_count=4,
        selection_seed=42000, training_seed=420000)

    rank_file = joinpath(output_directory, "rank_selection", "selected_ranks.toml")
    ranks = Dict{Int, Vector{Int}}()
    if isfile(rank_file)
        for (fold, selected_ranks) in TOML.parsefile(rank_file)["folds"]
            ranks[parse(Int, fold)] = Int.(selected_ranks)
        end
    else
        rank_configuration = merge(configuration(:full, :balanced, :uniform, :both, 0.0, nothing),
            (output_directory=joinpath(output_directory, "rank_selection"),))
        pilot_splits = filter(split -> split.repeat == 1, event_splits)
        ranks = LoRARankOptimisation.select_ranks(pretrained_model, datasets[:uniform], pilot_splits;
            settings=rank_configuration,
            pilot_training=merge(training_settings, (epochs=10,)),
            candidate_training=merge(training_settings,
                (epochs=50, validation_interval=1, patience=10)),
            thresholds=[0.8, 0.9, 0.95, 0.99],
            inner_split_settings=(repeats=collect(1:3), training_event_count=2,
                validation_event_count=2, holdout_seed=62000,
                selection_seed=63000, training_seed=640000))
    end

    jobs = NamedTuple[]
    for fold in 1:5
        fold_splits = filter(split -> split.fold == fold, event_splits)

        # Experiment 1:
        method_scores = NamedTuple[]
        for method in adaptation_methods
            validation_losses = Float64[]
            for split in fold_splits
                loss = run_condition!(jobs, pretrained_model, datasets, ranks,
                    1, method, :balanced, :uniform, :both, 0.0, split)
                push!(validation_losses, loss)
            end
            push!(method_scores, (candidate=method, validation=mean(validation_losses)))
        end
        sort!(method_scores; by=score -> (score.validation, string(score.candidate)))
        Surge.save_table(joinpath(output_directory, "selections", "method_fold_$fold.tsv"), method_scores)
        selected_method = first(method_scores).candidate

        # Experiment 2:
        objective_scores = NamedTuple[]
        for objective in selection_objectives
            validation_losses = Float64[]
            for split in fold_splits
                loss = run_condition!(jobs, pretrained_model, datasets, ranks,
                    2, selected_method, objective, :uniform, :both, 0.0, split)
                push!(validation_losses, loss)
            end
            push!(objective_scores, (candidate=objective, validation=mean(validation_losses)))
        end
        sort!(objective_scores; by=score -> (score.validation, string(score.candidate)))
        Surge.save_table(joinpath(output_directory, "selections", "temporal_fold_$fold.tsv"), objective_scores)
        selected_objective = first(objective_scores).candidate
        for objective in additional_objectives
            for split in fold_splits
                run_condition!(jobs, pretrained_model, datasets, ranks,
                    2, selected_method, objective, :uniform, :both, 0.0, split)
            end
        end

        # Experiment 3:
        for (profile, layout) in geometry_cases
            for split in fold_splits
                run_condition!(jobs, pretrained_model, datasets, ranks,
                    3, selected_method, selected_objective, profile, layout, 0.0, split)
            end
        end

        # Experiment 4:
        for noise_std in noise_standard_deviations
            for split in fold_splits
                run_condition!(jobs, pretrained_model, datasets, ranks,
                    4, selected_method, selected_objective, :uniform, :both, noise_std, split)
            end
        end
    end

    evaluated_directories = Set{String}()
    for job in jobs
        if job.directory in evaluated_directories
            continue
        end
        push!(evaluated_directories, job.directory)
        metrics_path = joinpath(job.directory, "diagnostics", "metrics.tsv")
        if !isfile(metrics_path)
            save_series = job.method == :full && job.profile == :uniform && job.layout == :both &&
                job.noise_std == 0.0 && job.temporal in (:balanced, :long, :one_step)
            ExperimentDiagnostics.diagnose_saved_run(job.directory, datasets[job.profile];
                pretrained_model=pretrained_model, forecast_hours=72.0,
                interior_indices=collect(9:92), interval_boundaries=[0.0, 12.0, 24.0, 48.0, 72.0],
                save_series=save_series)
        end
    end

    metrics = NamedTuple[]
    for job in jobs
        rows = Surge.load_table(joinpath(job.directory, "diagnostics", "metrics.tsv"))
        if job.stage == 1
            condition = string(job.method)
        elseif job.stage == 2
            condition = string(job.temporal)
        elseif job.stage == 3
            condition = "$(job.profile)__$(job.layout)"
        else
            condition = "noise$(round(Int, 100 * job.noise_std))cm"
        end
        for row in rows
            push!(metrics, (stage=job.stage, condition=condition, fold=job.fold,
                repeat=job.repeat, training_seconds=job.training_seconds, row...))
        end
    end
    Surge.save_table(joinpath(output_directory, "event_metrics.tsv"), metrics)

    columns = (:stage, :condition, :model, :from_hours, :to_hours, :events,
        :mse_h, :mse_u, :anomaly_mse, :bias_squared, :training_seconds)
    summaries = [NamedTuple{columns}(Tuple(getproperty(row, key) for key in columns))
        for row in ExperimentDiagnostics.summarize_run_diagnostics(metrics;
            group_by=(:stage, :condition, :model, :from_hours, :to_hours))]
    Surge.save_table(joinpath(output_directory, "summary.tsv"), summaries)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
