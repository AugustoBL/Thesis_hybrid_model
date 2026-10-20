# Shared imports and persistence only. Scientific settings stay in each runner.
# Include Surge in Main so that BSON can resolve the original model types.
# 1. Load the existing model, data, training and diagnostic functions ----------
include(joinpath(@__DIR__, "..", "src", "Surge.jl"))
# Use Surge.function_name below to make the implementation source easy to identify.
import .Surge
# Flux supplies Adam; JLD2 stores records; the remaining imports support sampling,
# summary statistics, file fingerprints, rank files and creation timestamps.
using Flux, JLD2, Random, Statistics, SHA, TOML, Dates
# Reuse the original dataset loader and source-pretraining routines.
include(joinpath(@__DIR__, "..", "thesis_experiments", "shared", "conservative_pilot_utils.jl"))
# Reuse the original free-forecast and spatial/temporal error calculations.
include(joinpath(@__DIR__, "..", "src", "experiment_diagnostics.jl"))

# src.fit_run does not save the original timed checkpoint envelope or timing
# fields. Keep that small piece of orchestration here, calling the src trainer.
# Inputs: common pretrained weights, complete trajectories, one fold/repeat split,
# that fit's settings, and its output directory. Return the saved result record.
function fit_timed_run(pretrained_model, dataset, split, configuration, directory)
    # 2. Reuse a completed fit or recover its latest resume state -------------
    # result.jld2 marks completion; its result key contains metadata, not model weights.
    result_path = joinpath(directory, "result.jld2")
    # A completed fit needs no new data preparation or optimiser steps.
    if isfile(result_path)
        return JLD2.load(result_path, "result")
    end
    # Create the run directory before saving checkpoints or exported models.
    mkpath(directory)
    # The checkpoint stores the optimiser moments and RNG as well as the current model.
    checkpoint_path = joinpath(directory, "checkpoint.jld2")
    # nothing means that the fit starts from the pretrained weights.
    previous = nothing
    # An interrupted fit supplies its saved training state and accumulated timings.
    if isfile(checkpoint_path)
        previous = JLD2.load(checkpoint_path, "saved")
    end

    # 3. Prepare observations, loss scales and the trainable model ------------
    # Convert the monotonic nanosecond clock to seconds for preparation timing.
    preparation_start = time_ns() / 1e9
    # Select split.train/split.validation events, truncate their training records,
    # and extract sparse h observations with the configured independent noise seeds.
    # h denotes water level [m]; the model also evolves velocity u [m/s], which is not supervised.
    training_data, validation_data = Surge.prepare_training_split(dataset, split, configuration)
    # Measure pretrained MSE at each training lead; these fixed scales balance the losses.
    # Only training data determine the scales, so validation/test errors cannot influence them.
    normalizers = Surge.compute_pretrained_error_scales(pretrained_model, training_data,
        configuration.training, configuration.stations)
    # Only LoRA needs a vector of per-layer ranks; other methods leave this unused.
    ranks = nothing
    if configuration.adaptation.method == :lora
        # Use ranks selected inside this outer fold's training pool.
        ranks = configuration.adaptation.ranks[split.fold]
    end
    # Start an independent model with the chosen trainable layers, factors or FiLM parameters.
    # The split seed makes any adaptation-specific initialisation reproducible.
    # LoRA trains low-rank weight factors plus biases; FiLM trains feature scales and shifts.
    model = Surge.configure_adaptation(pretrained_model, configuration.adaptation.method;
        seed=split.seed, ranks=ranks)
    # Stop preparation timing before entering the training loop.
    preparation_seconds = time_ns() / 1e9 - preparation_start
    # A fresh fit has no previous training time and no optimiser/RNG state to restore.
    previous_training_seconds = 0.0
    resume_state = nothing
    # For a resumed fit, add earlier timings and let the shared trainer restore its state.
    if previous !== nothing
        preparation_seconds += previous.preparation_seconds
        previous_training_seconds = previous.training_seconds
        resume_state = previous.state
    end

    # 4. Train and save resume checkpoints at the trainer's validation epochs -
    # This timer includes state refresh, optimisation, validation and checkpoint saving.
    training_start = time_ns() / 1e9
    # The trainer calls this function with its current model, optimiser, RNG and history.
    save_checkpoint = function(state)
        # Preserve the original saved envelope and cumulative timing fields.
        saved = (state=state, preparation_seconds=preparation_seconds,
            training_seconds=previous_training_seconds + time_ns() / 1e9 - training_start)
        # Write a complete temporary JLD2 file before replacing the last checkpoint.
        JLD2.jldsave(checkpoint_path * ".tmp"; saved=saved)
        mv(checkpoint_path * ".tmp", checkpoint_path; force=true)
        # Export the same learning-curve records as a readable table.
        Surge.save_table(joinpath(directory, "training_history.tsv"), state.history)
    end
    # The shared loop refreshes model-generated window states, shuffles batches,
    # applies the chosen rollout loss, and selects the lowest sparse validation MSE.
    # resume restores the previous epoch/optimiser/RNG instead of restarting the fit.
    state = Surge.train_model!(model, training_data, validation_data;
        settings=configuration.training, stations=configuration.stations,
        normalizers=normalizers,
        # Construct Adam with the original learning rate, moment decays and epsilon.
        optimizer=Adam(configuration.optimizer.learning_rate,
            configuration.optimizer.betas, configuration.optimizer.epsilon),
        # This RNG controls training order and is also retained in resume checkpoints.
        rng=MersenneTwister(split.seed + configuration.optimizer_seed_offset),
        checkpoint=save_checkpoint, resume=resume_state)
    # Add this invocation's training duration to any earlier interrupted invocation.
    training_seconds = previous_training_seconds + time_ns() / 1e9 - training_start

    # 5. Export final/selected weights and the original result record ----------
    # Save the last iterate before restoring the validation-selected parameters.
    Surge.save_model(joinpath(directory, "final_model.bson"), state.model)
    # Replace only trainable arrays; frozen pretrained arrays already have their correct values.
    Surge.restore_trainable_parameters!(state.model, state.best_parameters)
    # Forecast evaluation loads this validation-selected model, rather than the last iterate.
    Surge.save_model(joinpath(directory, "best_model.bson"), state.model)
    # Retain settings, event memberships, loss history and checkpoint-selection details.
    # Model weights live in the BSON files; normalizers retain the fixed per-lead MSE scales.
    result = (configuration=configuration, split=split, history=state.history,
        best_validation_loss=state.best_loss, best_epoch=state.best_epoch,
        completed_epochs=state.epoch, updates=state.updates,
        trainable_parameters=Surge.trainable_parameter_count(state.model),
        normalizers=normalizers, training_seconds=training_seconds,
        preparation_seconds=preparation_seconds)
    # Publish the completion record only after both model files have been saved.
    JLD2.jldsave(result_path * ".tmp"; result=result)
    mv(result_path * ".tmp", result_path; force=true)
    # The runners use this metadata for validation selection and their run-index tables.
    return result
end
