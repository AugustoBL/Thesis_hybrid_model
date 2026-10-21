module LoRARankOptimisation

using ..Surge
using LinearAlgebra, Random, Statistics, JLD2, TOML

function kernel_update_spectra(pretrained_model, fitted_model)
    spectra = NamedTuple[]
    for (pretrained_layer, fitted_layer) in zip(pretrained_model.f.layers, fitted_model.f.layers) # Pair corresponding initial and fitted kernels.
        update = reshape(Float64.(fitted_layer.weight .- pretrained_layer.weight), :, size(pretrained_layer.weight, 3)) #  Delta W = W_fit - W_0 as a (kernel_width * input_channels)
        decomposition = svd(update) # Factor Delta W = U * Diagonal(sigma) * V',
        push!(spectra, (update=update, singular_values=decomposition.S))
    end
    return spectra
end

function rank_for_energy(singular_values, threshold) # Choose the smallest rank retaining the requested fraction of squared update norm.
    energy = sum(abs2, singular_values) # E = sum_i sigma_i^2 = ||Delta W||_F^2
    if energy == 0
        return 0
    end
    explained_energy = cumsum(singular_values.^2) ./ energy # E_r / E = sum_{i=1}^r sigma_i^2 / sum_i sigma_i^2.
    selected_rank = findfirst(fraction -> fraction >= threshold, explained_energy) # Find min r with E_r/E >= threshold
    if selected_rank === nothing
        return length(singular_values) # Retain every available singular direction.
    end
    return selected_rank
end

"""Paired inner-validation selection; one-standard-error rule with repeated draws."""
function select_candidate(candidates)
    validation_means = [mean(candidate.validation_losses) for candidate in candidates] # For each candidate c, compute mean lossacross repeats.
    best_index = first(sortperm(eachindex(candidates); by=index ->
        (validation_means[index], candidates[index].parameter_count, candidates[index].threshold))) # Minimize mean loss first, then parameter count, then energy threshold to break ties.
    repeat_count = length(candidates[best_index].validation_losses) # R is the number of repeated fits.
    standard_error = 0.0
    if repeat_count > 1
        standard_error = std(candidates[best_index].validation_losses) / sqrt(repeat_count) # SE = s_best/sqrt(R), using sample standard deviation
    end
    cutoff = validation_means[best_index] + standard_error # Accept mean losses up to Lbar_best + SE_best.
    eligible_indices = findall(loss -> loss <= cutoff, validation_means) # Build the set of candidates inside the one-SE tolerance.
    selected_index = first(sort(eligible_indices; by=index ->
        (candidates[index].parameter_count, validation_means[index], candidates[index].threshold))) # Minimize trainable parameter count, then mean loss, then threshold.
    return (selected_index=selected_index, best_index=best_index,
        validation_means=validation_means, standard_error=standard_error, cutoff=cutoff)
end

function inner_event_splits(outer_split; repeats, training_event_count, validation_event_count,
                            holdout_seed, selection_seed, training_seed)
    shuffled_pool = shuffle(MersenneTwister(holdout_seed + outer_split.fold), outer_split.pool)
    validation_events = sort(shuffled_pool[1:validation_event_count])
    training_pool = sort(setdiff(outer_split.pool, validation_events)) # Remove inner-validation events so no candidate trains on its selection data.
    splits = NamedTuple[]
    for repeat in repeats
        random_generator = MersenneTwister(selection_seed + 100 * outer_split.fold + repeat)
        training_events = sort(shuffle(random_generator, training_pool)[1:training_event_count])
        push!(splits, (fold=outer_split.fold, repeat=repeat, train=training_events,
            validation=validation_events, test=outer_split.test, pool=training_pool,
            seed=training_seed + 100 * outer_split.fold + repeat)) # Assign the fitting seed shared by candidates for this fold/repeat.
    end
    return splits
end

function rank_fit_configuration(settings, training, method, ranks, name)
    return (experiment=1, name=name, output_directory=settings.output_directory,
        dataset_file=settings.dataset_file, pretrained_file=settings.pretrained_file,
        measurement_gap=settings.measurement_gap, training_steps=settings.training_steps,
        stations=settings.stations, observation_noise_std=settings.observation_noise_std,
        training_noise_seed_offset=settings.training_noise_seed_offset,
        validation_noise_seed_offset=settings.validation_noise_seed_offset,
        optimizer_seed_offset=settings.optimizer_seed_offset, optimizer=settings.optimizer,
        adaptation=(method=method, ranks=ranks), training=training)
end

"""Select per-layer LoRA ranks within each outer-training pool."""
function select_ranks(pretrained_model, dataset, outer_splits; settings, pilot_training,
                      candidate_training, thresholds, inner_split_settings)
    selected_ranks = Dict{Int,Vector{Int}}() # Map each fold f to its selected vector (r_1, ..., r_L).
    for outer_split in outer_splits
        fold_directory = joinpath(settings.output_directory, "fold_$(outer_split.fold)")
        mkpath(fold_directory)
        inner_splits = inner_event_splits(outer_split; inner_split_settings...) # Create shared inner-validation data and repeat-specific training draws.
        pilot_configuration = rank_fit_configuration(settings, pilot_training, :full, nothing,
            "preliminary_full_fit")
        pilot_run = only(Surge.run_experiment(pretrained_model, dataset, inner_splits[1:1]; # Fit or reload exactly one pilot, using only the first inner repeat.
            configuration=pilot_configuration, save_final=true))
        pilot_model = Surge.load_model(joinpath(pilot_run.directory, "final_model.bson"))
        spectra = kernel_update_spectra(pretrained_model, pilot_model) # Compute singular values of each reshaped Delta W_l = W_fit,l - W_0,l.
        JLD2.jldsave(joinpath(fold_directory, "update_spectra.jld2"); spectra=spectra) # Save the update matrices and spectra
        candidates = NamedTuple[]
        for threshold in thresholds # Test each requested retained-energy fraction q.
            ranks = [rank_for_energy(spectrum.singular_values, threshold) for spectrum in spectra]
            ranks_by_fold = Dict(outer_split.fold => ranks)
            configuration = rank_fit_configuration(settings, candidate_training, :lora, ranks_by_fold,
                "candidate_" * join(ranks, "_"))
            runs = Surge.run_experiment(pretrained_model, dataset, inner_splits; configuration=configuration)
            losses = [run.result.best_validation_loss for run in runs]
            push!(candidates, (threshold=threshold, ranks=ranks, validation_losses=losses,
                parameter_count=first(runs).result.trainable_parameters))
        end
        decision = select_candidate(candidates) # Prefer the smallest parameter count within one SE of the best mean validation loss.
        selected = candidates[decision.selected_index] # Retrieve the selected candidate's ranks and statistics.
        selected_ranks[outer_split.fold] = selected.ranks # Assign this fold's selected layerwise ranks for later independent final fits.
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
    open(joinpath(settings.output_directory, "selected_ranks.toml"), "w") do output
        TOML.print(output, Dict("folds" => Dict(string(fold) => ranks for (fold, ranks) in selected_ranks)))
    end
    return selected_ranks
end

end
