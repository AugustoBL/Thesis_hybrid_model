module LoRARankOptimisation

using ..Surge
using LinearAlgebra, Random, Statistics, JLD2, TOML

"""
Compute singular values of the kernel changes from a fully fitted pilot model.

Args:
    pretrained_model: Base wrapper with convolution kernels `(kernel_width, input_channels, output_channels)`.
    fitted_model: Fully fitted wrapper with corresponding convolution layers.

Returns:
    One record per layer: Float64 update `(kernel_width * input_channels, output_channels)`
    and singular values in descending order. Bias changes are excluded.
"""
function kernel_update_spectra(pretrained_model, fitted_model)
    spectra = NamedTuple[]
    # Pair each fitted kernel with the corresponding pretrained kernel.
    for (pretrained_layer, fitted_layer) in zip(pretrained_model.f.layers, fitted_model.f.layers)
        # Use (kernel width * input channels, output channels) to match the LoRA factors.
        update = reshape(Float64.(fitted_layer.weight .- pretrained_layer.weight), :, size(pretrained_layer.weight, 3))
        # Use SVD to measure the energy in each update direction.
        decomposition = svd(update)
        push!(spectra, (update=update, singular_values=decomposition.S))
    end
    return spectra
end

"""
Choose the smallest rank keeping a requested fraction of squared kernel-update norm.

Args:
    singular_values: Descending singular values of a layer's kernel-update matrix.
    threshold: Requested retained-energy fraction, normally between zero and one.

Returns:
    Retained rank. Zero for a zero update, or full available rank if the threshold is not reached.
"""
function rank_for_energy(singular_values, threshold)
    # Sum of squared singular values equals the squared Frobenius norm of the update.
    energy = sum(abs2, singular_values)
    if energy == 0
        return 0
    end
    # Each entry is the fraction retained by the first r singular directions.
    explained_energy = cumsum(singular_values.^2) ./ energy
    selected_rank = findfirst(fraction -> fraction >= threshold, explained_energy)
    if selected_rank === nothing
        # Use full rank if the threshold is not reached.
        return length(singular_values)
    end
    return selected_rank
end

"""
Select a candidate using the one-standard-error rule on shared inner splits.

Args:
    candidates: Records with validation_losses (one per repeat), parameter_count and threshold.

Returns:
    Selected/best indices, mean validation losses, the best candidate's standard error
    and acceptance cutoff. Selects the smallest parameter count within that cutoff.
"""
function select_candidate(candidates)
    # Compare candidates by mean loss on the same repeated training/validation splits.
    validation_means = [mean(candidate.validation_losses) for candidate in candidates]
    # Resolve equal mean losses by parameter count, then energy threshold.
    best_index = first(sortperm(eachindex(candidates); by=index ->
        (validation_means[index], candidates[index].parameter_count, candidates[index].threshold)))
    repeat_count = length(candidates[best_index].validation_losses)
    standard_error = 0.0
    if repeat_count > 1
        # Estimate the standard error across repeats. One repeat gives zero tolerance.
        standard_error = std(candidates[best_index].validation_losses) / sqrt(repeat_count)
    end
    cutoff = validation_means[best_index] + standard_error
    eligible_indices = findall(loss -> loss <= cutoff, validation_means)
    # Prefer fewer parameters among eligible candidates, then lower loss, then lower threshold.
    selected_index = first(sort(eligible_indices; by=index ->
        (candidates[index].parameter_count, validation_means[index], candidates[index].threshold)))
    return (selected_index=selected_index, best_index=best_index,
        validation_means=validation_means, standard_error=standard_error, cutoff=cutoff)
end

"""
Create shared inner-validation events and repeat-specific training draws.

Args:
    outer_split: Outer fold, available training pool and held-out test-event indices.
    repeats: Repeat identifiers.
    training_event_count: Events sampled for each inner training fit.
    validation_event_count: Events held out for inner validation.
    holdout_seed: Base seed selecting fixed validation events within each fold.
    selection_seed: Base seed selecting repeated training subsets.
    training_seed: Base seed for model initialization and optimization.

Returns:
    Split records with disjoint inner training/validation events and unchanged outer test events.
"""
function inner_event_splits(outer_split; repeats, training_event_count, validation_event_count,
                            holdout_seed, selection_seed, training_seed)
    # Select validation only from the outer training pool, never from outer test events.
    shuffled_pool = shuffle(MersenneTwister(holdout_seed + outer_split.fold), outer_split.pool)
    validation_events = sort(shuffled_pool[1:validation_event_count])
    # Exclude validation events from every candidate's training pool.
    training_pool = sort(setdiff(outer_split.pool, validation_events))
    splits = NamedTuple[]
    for repeat in repeats
        # All candidates receive the same event draws and fitting seed for this fold/repeat.
        random_generator = MersenneTwister(selection_seed + 100 * outer_split.fold + repeat)
        training_events = sort(shuffle(random_generator, training_pool)[1:training_event_count])
        push!(splits, (fold=outer_split.fold, repeat=repeat, train=training_events,
            validation=validation_events, test=outer_split.test, pool=training_pool,
            seed=training_seed + 100 * outer_split.fold + repeat))
    end
    return splits
end

"""
Build an experiment configuration for a pilot or LoRA candidate fit.

Args:
    settings: Dataset/model paths, observations, optimizer, seeds and output directory.
    training: Training settings for this fit.
    method: Adaptation method (:full for the pilot, :lora for candidates).
    ranks: Fold-to-layer-ranks dictionary for LoRA, or nothing for the pilot.
    name: Run name used to identify the fit.

Returns:
    Configuration record accepted by Surge.run_experiment.
"""
function rank_fit_configuration(settings, training, method, ranks, name)
    # Keep shared data and optimizer settings and change the fit settings.
    return (experiment=1, name=name, output_directory=settings.output_directory,
        dataset_file=settings.dataset_file, pretrained_file=settings.pretrained_file,
        measurement_gap=settings.measurement_gap, training_steps=settings.training_steps,
        stations=settings.stations, observation_noise_std=settings.observation_noise_std,
        training_noise_seed_offset=settings.training_noise_seed_offset,
        validation_noise_seed_offset=settings.validation_noise_seed_offset,
        optimizer_seed_offset=settings.optimizer_seed_offset, optimizer=settings.optimizer,
        adaptation=(method=method, ranks=ranks), training=training)
end

"""
Select per-layer LoRA ranks using a full-fit pilot and inner validation.

Args:
    pretrained_model: Base convolutional surrogate.
    dataset: Event ensemble used for inner fits.
    outer_splits: Outer splits providing fold identifiers and available training pools.
    settings: Common experiment paths, observations, optimizer and seeds.
    pilot_training: Training settings for the full-adaptation pilot.
    candidate_training: Training settings for LoRA candidate fits.
    thresholds: Retained kernel-update energy fractions defining candidate ranks.
    inner_split_settings: Counts, repeats and seeds forwarded to inner_event_splits.

Returns:
    Dictionary mapping outer fold to a rank vector (one entry per convolution).
    Saves pilot spectra, candidate decisions, inner splits and selected-rank reports.
"""
function select_ranks(pretrained_model, dataset, outer_splits; settings, pilot_training,
                      candidate_training, thresholds, inner_split_settings)
    selected_ranks = Dict{Int,Vector{Int}}()
    for outer_split in outer_splits
        fold_directory = joinpath(settings.output_directory, "fold_$(outer_split.fold)")
        mkpath(fold_directory)
        # Reuse identical inner splits across all thresholds for this outer fold.
        inner_splits = inner_event_splits(outer_split; inner_split_settings...)
        pilot_configuration = rank_fit_configuration(settings, pilot_training, :full, nothing,
            "preliminary_full_fit")
        # Fit or reload one full-adaptation pilot using the first inner repeat only.
        pilot_run = only(Surge.run_experiment(pretrained_model, dataset, inner_splits[1:1];
            configuration=pilot_configuration, save_final=true))
        # Use the final pilot parameters for spectra instead of the best validation model.
        pilot_model = Surge.load_model(joinpath(pilot_run.directory, "final_model.bson"))
        spectra = kernel_update_spectra(pretrained_model, pilot_model)
        JLD2.jldsave(joinpath(fold_directory, "update_spectra.jld2"); spectra=spectra)
        candidates = NamedTuple[]
        # Convert each energy threshold into layer-wise ranks and fit on all inner repeats.
        for threshold in thresholds
            ranks = [rank_for_energy(spectrum.singular_values, threshold) for spectrum in spectra]
            ranks_by_fold = Dict(outer_split.fold => ranks)
            configuration = rank_fit_configuration(settings, candidate_training, :lora, ranks_by_fold,
                "candidate_" * join(ranks, "_"))
            runs = Surge.run_experiment(pretrained_model, dataset, inner_splits; configuration=configuration)
            # Compare best validation losses. Count trainable biases as parameters.
            losses = [run.result.best_validation_loss for run in runs]
            push!(candidates, (threshold=threshold, ranks=ranks, validation_losses=losses,
                parameter_count=first(runs).result.trainable_parameters))
        end
        # Apply the one-standard-error rule and retain ranks for later outer-fold fits.
        decision = select_candidate(candidates)
        selected = candidates[decision.selected_index]
        selected_ranks[outer_split.fold] = selected.ranks
        # Save the selected ranks and the data used to choose them.
        report = Dict("ranks" => selected.ranks, "threshold" => selected.threshold,
            "standard_error" => decision.standard_error, "validation_cutoff" => decision.cutoff,
            "criterion" => "one standard error; minimum validation loss for a single repeat",
            "candidates" => [Dict("threshold" => candidate.threshold, "ranks" => candidate.ranks,
                "validation_losses" => candidate.validation_losses,
                "parameter_count" => candidate.parameter_count) for candidate in candidates])
        open(joinpath(fold_directory, "selected_ranks.toml"), "w") do output
            TOML.print(output, report)
        end
        JLD2.jldsave(joinpath(fold_directory, "selection.jld2"); candidates=candidates,
            decision=decision, inner_splits=inner_splits, outer_split=outer_split)
    end
    # Save the selected ranks for all folds in one file.
    open(joinpath(settings.output_directory, "selected_ranks.toml"), "w") do output
        TOML.print(output, Dict("folds" => Dict(string(fold) => ranks for (fold, ranks) in selected_ranks)))
    end
    return selected_ranks
end

end
