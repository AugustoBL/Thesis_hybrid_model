module LoRARankOptimisation # Isolate the preliminary rank-selection calculations from the main experiment.

using ..Surge # Access the parent project's model fitting and checkpoint-loading functions.
using LinearAlgebra, Random, Statistics, JLD2, TOML # Supply SVD, seeded sampling, mean/std, and binary/text persistence.

function kernel_update_spectra(pretrained_model, fitted_model) # Measure the matrix rank structure of each full-fit kernel change.
    spectra = NamedTuple[] # Collect one update matrix and singular-value vector per layer; biases are excluded.
    for (pretrained_layer, fitted_layer) in zip(pretrained_model.f.layers, fitted_model.f.layers) # Pair corresponding initial and fitted kernels.
        update = reshape(Float64.(fitted_layer.weight .- pretrained_layer.weight), :, size(pretrained_layer.weight, 3)) # Form Delta W = W_fit - W_0 as a (kernel_width * input_channels) by output_channels matrix, then use Float64 for SVD.
        decomposition = svd(update) # Factor Delta W = U * Diagonal(sigma) * V', with singular values in decreasing order.
        push!(spectra, (update=update, singular_values=decomposition.S)) # Retain Delta W and sigma; sum(sigma.^2) equals its squared Frobenius norm.
    end # Finish measuring all layer updates.
    return spectra # Return layerwise spectra in network order.
end # End kernel_update_spectra.

function rank_for_energy(singular_values, threshold) # Choose the smallest rank retaining the requested fraction of squared update norm.
    energy = sum(abs2, singular_values) # E = sum_i sigma_i^2 = ||Delta W||_F^2; this is matrix energy, not physical energy.
    if energy == 0 # A zero update has no nonzero singular directions and would otherwise cause division by zero.
        return 0 # No kernel factors are needed; the project's rank-zero LoRA still permits bias adaptation.
    end # Continue only for a nonzero update.
    explained_energy = cumsum(singular_values.^2) ./ energy # E_r / E = sum_{i=1}^r sigma_i^2 / sum_i sigma_i^2.
    selected_rank = findfirst(fraction -> fraction >= threshold, explained_energy) # Find min r with E_r/E >= threshold; the best rank-r approximation leaves squared residual E - E_r.
    if selected_rank === nothing # Fall back if no cumulative fraction reaches the supplied threshold.
        return length(singular_values) # Retain every available singular direction.
    end # A qualifying rank was found.
    return selected_rank # Return the minimum qualifying rank, assuming descending singular values as supplied by SVD.
end # End rank_for_energy.

"""Paired inner-validation selection; one-standard-error rule with repeated draws."""
function select_candidate(candidates) # Apply a one-standard-error complexity preference to paired repeated validation fits.
    validation_means = [mean(candidate.validation_losses) for candidate in candidates] # For each candidate c, compute mean loss Lbar_c across repeats.
    best_index = first(sortperm(eachindex(candidates); by=index -> # Identify the candidate with the lexicographically smallest selection key.
        (validation_means[index], candidates[index].parameter_count, candidates[index].threshold))) # Minimize mean loss first, then parameter count, then energy threshold to break ties.
    repeat_count = length(candidates[best_index].validation_losses) # R is the number of repeated fits for the minimum-mean-loss candidate.
    standard_error = 0.0 # With one repeat, use zero tolerance rather than estimating an undefined sample standard deviation.
    if repeat_count > 1 # Estimate dispersion only when at least two repeated losses are available.
        standard_error = std(candidates[best_index].validation_losses) / sqrt(repeat_count) # SE = s_best/sqrt(R), using sample standard deviation; this is a selection heuristic, not a held-out confidence interval.
    end # Finish the best candidate's standard-error estimate; no paired-difference SE is computed here.
    cutoff = validation_means[best_index] + standard_error # Accept mean losses up to Lbar_best + SE_best.
    eligible_indices = findall(loss -> loss <= cutoff, validation_means) # Build the set of candidates inside the one-SE tolerance.
    selected_index = first(sort(eligible_indices; by=index -> # Choose the least complex member of that eligible set.
        (candidates[index].parameter_count, validation_means[index], candidates[index].threshold))) # Minimize trainable parameter count, then mean loss, then threshold.
    return (selected_index=selected_index, best_index=best_index, # Report both the complexity-selected candidate and the minimum-loss candidate.
        validation_means=validation_means, standard_error=standard_error, cutoff=cutoff) # Include the losses and tolerance needed to audit the decision.
end # End select_candidate.

function inner_event_splits(outer_split; repeats, training_event_count, validation_event_count, # Construct inner data partitions entirely within the outer training pool.
                            holdout_seed, selection_seed, training_seed) # Separate seeds control validation selection, repeated training draws, and fitting.
    shuffled_pool = shuffle(MersenneTwister(holdout_seed + outer_split.fold), outer_split.pool) # Generate a reproducible random permutation of the fold's eligible events.
    validation_events = sort(shuffled_pool[1:validation_event_count]) # Select a fixed inner-validation subset without replacement; sorting only fixes presentation order.
    training_pool = sort(setdiff(outer_split.pool, validation_events)) # Remove inner-validation events so no candidate trains on its selection data.
    splits = NamedTuple[] # Collect one train/validation/test record per repeat.
    for repeat in repeats # Reuse identical repeat-specific splits across all candidate ranks.
        random_generator = MersenneTwister(selection_seed + 100 * outer_split.fold + repeat) # Seed the training-subset permutation deterministically by fold and repeat.
        training_events = sort(shuffle(random_generator, training_pool)[1:training_event_count]) # Draw the requested number of distinct training events from the eligible pool.
        push!(splits, (fold=outer_split.fold, repeat=repeat, train=training_events, # Record this repeated training draw and its outer-fold identity.
            validation=validation_events, test=outer_split.test, pool=training_pool, # Keep inner validation fixed and carry outer-test IDs as metadata, not selection losses.
            seed=training_seed + 100 * outer_split.fold + repeat)) # Assign the fitting seed shared by candidates for this fold/repeat.
    end # Finish constructing the paired repeated draws.
    return splits # Return partitions with disjoint inner training and validation events.
end # End inner_event_splits.

function rank_fit_configuration(settings, training, method, ranks, name) # Assemble the existing fitter's inputs without changing its objective or optimizer.
    return (experiment=1, name=name, output_directory=settings.output_directory, # Set the experiment identity and location of fitted-model artifacts.
        dataset_file=settings.dataset_file, pretrained_file=settings.pretrained_file, # Identify the event ensemble and common starting weights W_0.
        measurement_gap=settings.measurement_gap, training_steps=settings.training_steps, # Specify observation spacing in model steps and the available training record length.
        stations=settings.stations, observation_noise_std=settings.observation_noise_std, # Fix the observed water-level cells and observation-noise standard deviation.
        training_noise_seed_offset=settings.training_noise_seed_offset, # Preserve reproducible training-observation noise for paired comparisons.
        validation_noise_seed_offset=settings.validation_noise_seed_offset, # Preserve reproducible validation-observation noise separately from training noise.
        optimizer_seed_offset=settings.optimizer_seed_offset, optimizer=settings.optimizer, # Copy optimizer settings and the seed offset controlling stochastic fitting.
        adaptation=(method=method, ranks=ranks), training=training) # Select full adaptation or layerwise rank-r LoRA updates and the supplied rollout/loss schedule.
end # End rank_fit_configuration.

"""Standalone preliminary fitting and rank selection. Never included by the main pipeline."""
function select_ranks(pretrained_model, dataset, outer_splits; settings, pilot_training, # Select a separate layerwise rank vector for each outer fold without using its test losses.
                      candidate_training, thresholds, inner_split_settings) # Receive pilot/candidate training budgets, energy thresholds, and inner-split rules.
    selected_ranks = Dict{Int,Vector{Int}}() # Map each fold f to its selected vector (r_1, ..., r_L).
    for outer_split in outer_splits # Process each supplied outer-split record; callers should provide one record per fold.
        fold_directory = joinpath(settings.output_directory, "fold_$(outer_split.fold)") # Separate artifacts by outer fold.
        mkpath(fold_directory) # Ensure the artifact directory exists; this performs no model calculation.
        inner_splits = inner_event_splits(outer_split; inner_split_settings...) # Create shared inner-validation data and repeat-specific training draws.
        pilot_configuration = rank_fit_configuration(settings, pilot_training, :full, nothing, # Configure a full-parameter pilot whose kernel updates suggest candidate ranks.
            "preliminary_full_fit") # Name the pilot separately from the low-rank candidate fits.
        pilot_run = only(Surge.run_experiment(pretrained_model, dataset, inner_splits[1:1]; # Fit or reload exactly one pilot, using only the first inner repeat.
            configuration=pilot_configuration)) # Use the unchanged experiment runner and require a single returned run.
        # Rank estimation uses FINAL full-fit updates, not the validation-best weights.
        pilot_model = Surge.load_model(joinpath(pilot_run.directory, "final_model.bson")) # Load the final pilot iterate W_fit, rather than its validation-best iterate.
        spectra = kernel_update_spectra(pretrained_model, pilot_model) # Compute singular values of each reshaped Delta W_l = W_fit,l - W_0,l.
        JLD2.jldsave(joinpath(fold_directory, "update_spectra.jld2"); spectra=spectra) # Save the update matrices and spectra for inspection without recomputing the pilot.
        candidates = NamedTuple[] # Accumulate rank vectors, repeated validation losses, and parameter counts.
        for threshold in thresholds # Test each requested retained-energy fraction q.
            ranks = [rank_for_energy(spectrum.singular_values, threshold) for spectrum in spectra] # Set each r_l to the minimum rank retaining fraction q of that layer's squared update norm.
            ranks_by_fold = Dict(outer_split.fold => ranks) # Package the rank vector in the fold-indexed format expected by LoRA configuration.
            configuration = rank_fit_configuration(settings, candidate_training, :lora, ranks_by_fold, # Configure fresh LoRA fitting with these ranks and the common candidate budget.
                "candidate_" * join(ranks, "_")) # Identify candidates by rank vector so thresholds producing identical ranks can reuse completed fits.
            # Every candidate restarts from pretrained weights with random/zero factors.
            # Equal rank vectors share the same completed runs and paired inner draws.
            runs = Surge.run_experiment(pretrained_model, dataset, inner_splits; configuration=configuration) # Fit/reload this candidate on every paired inner repeat, starting from pretrained weights rather than pilot weights.
            losses = [run.result.best_validation_loss for run in runs] # Use each repeat's minimum validation loss over its candidate-training epochs.
            push!(candidates, (threshold=threshold, ranks=ranks, validation_losses=losses, # Store q, the layerwise ranks, and the repeated selection losses.
                parameter_count=first(runs).result.trainable_parameters)) # Record actual trainable size, including biases; a layer's LoRA count is r*(kernel_width*input_channels + output_channels) + output_channels.
        end # Finish evaluating all threshold-derived candidates.
        decision = select_candidate(candidates) # Prefer the smallest parameter count within one SE of the best mean validation loss.
        selected = candidates[decision.selected_index] # Retrieve the selected candidate's ranks and statistics.
        selected_ranks[outer_split.fold] = selected.ranks # Assign this fold's selected layerwise ranks for later independent final fits.
        report = Dict("ranks" => selected.ranks, "threshold" => selected.threshold, # Begin a human-readable record of the chosen ranks and their originating energy threshold.
            "standard_error" => decision.standard_error, "validation_cutoff" => decision.cutoff, # Record SE_best and Lbar_best + SE_best, not the selected candidate's SE.
            "criterion" => "one standard error; minimum validation loss for a single repeat", # State the selection rule and its zero-SE single-repeat special case.
            "candidates" => [Dict("threshold" => candidate.threshold, "ranks" => candidate.ranks, # Include every candidate so the complexity/validation trade-off can be reconstructed.
                "validation_losses" => candidate.validation_losses, # Preserve individual repeated losses, not only their mean.
                "parameter_count" => candidate.parameter_count) for candidate in candidates]) # Pair each loss vector with its actual trainable parameter count.
        open(joinpath(fold_directory, "selected_ranks.toml"), "w") do output # Open this fold's selection report for replacement with the current decision.
            TOML.print(output, report) # Serialize the readable rank-selection report.
        end # Close the fold report file.
        JLD2.jldsave(joinpath(fold_directory, "selection.jld2"); candidates=candidates, # Preserve complete candidate records in binary form.
            decision=decision, inner_splits=inner_splits, outer_split=outer_split) # Save the rule outputs and exact event partitions for auditing leakage and pairing.
    end # Finish fold-specific selection.
    open(joinpath(settings.output_directory, "selected_ranks.toml"), "w") do output # Open the combined rank file consumed by later experiment configuration.
        TOML.print(output, Dict("folds" => Dict(string(fold) => ranks for (fold, ranks) in selected_ranks))) # Serialize the mapping f -> (r_1, ..., r_L), using string fold keys required by TOML.
    end # Close the combined rank file.
    return selected_ranks # Return the selected fold-to-rank mapping without launching final campaign fits.
end # End select_ranks.

end # End module LoRARankOptimisation.
