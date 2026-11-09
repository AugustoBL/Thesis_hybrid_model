# Loaded inside Surge after training_utils.jl, which defines the shared helpers.

"""
Compute average one-step prediction loss over batches.

Args:
    model: Model predicting the next state from an input.
    data: Input and target arrays with samples along the third dimension.
    batch_size: Maximum number of samples per batch.

Returns:
    Mean endpoint loss across all samples, weighted by actual batch size.
"""
function one_step_loss(model, data, batch_size)
    total = 0.0
    sample_count = size(data.input, 3)

    for first_index in 1:batch_size:sample_count
        last_index = min(first_index + batch_size - 1, sample_count)
        indices = first_index:last_index
        current_batch_size = length(indices)

        inputs = data.input[:, :, indices]
        targets = data.target[:, :, indices]

        predictions = model(inputs)
        batch_loss = endpoint_loss(predictions, targets, 1)

        # Weight by the actual number of samples
        total += current_batch_size * batch_loss
    end
    average_loss = total / sample_count
    return average_loss
end

"""
Add random noise to water levels and interior velocities.

Args:
    input: Input array with dimensions (space, features, batch).
    level: Noise amplitude. Zero returns the original input unchanged.
    rng: Random generator used to sample Gaussian noise.

Returns:
    Copy with zero-mean spatial water-level noise and unchanged boundary velocities.
    The final water-level row is padding and remains unchanged.
"""
function perturb_state_input(input, level, rng)
    level == 0 && return input
    perturbed = copy(input)
    noise_h = level .* randn(rng, Float32, size(input, 1)-1, 1, size(input, 3))
    noise_h .-= mean(noise_h; dims=1) # Do not change the volume the target must conserve.
    perturbed[1:end-1, 1:1, :] .+= noise_h
    perturbed[2:end-1, 2:2, :] .+= level .* randn(rng, Float32, size(input, 1)-2, 1, size(input, 3))
    return perturbed
end

"""
Train a model using noisy one-step inputs and clean targets.

Args:
    model: Model predicting state `(space, 2, batch)` from input features.
    training: Input `(space, features, samples)` and target `(space, 2, samples)` arrays.
    validation: Clean input/target arrays with the same layout as training.
    settings: Epochs, batching, learning-rate schedule, noise, seed and checkpoint interval.
    checkpoint: Callback receiving training state. Defaults to no saving.

Returns:
    Current model/optimizer/RNG state, history and best parameter snapshot, loss and epoch.
"""
function train_model!(model, training::NamedTuple, validation::NamedTuple; settings,
                      checkpoint=state -> nothing)

    optimizer_state = Flux.setup(Adam(settings.learning_rate), model)
    rng = MersenneTwister(settings.seed)
    best_loss = one_step_loss(model, validation, settings.batch_size)
    isfinite(best_loss) || error("Nonfinite initial validation loss")
    best_parameters, best_epoch = copy_trainable_parameters(model), 0
    history = NamedTuple[(epoch=0, learning_rate=settings.learning_rate,
        training_loss=NaN, validation_loss=best_loss)]


    for epoch in 1:settings.epochs
        completed_epochs = epoch - 1

        stage = fld(completed_epochs, settings.learning_rate_steps)

        if stage > settings.learning_rate_decay_stages
            stage = settings.learning_rate_decay_stages
        end
        rate = settings.learning_rate_decay_stages == 0 ? settings.learning_rate :
            settings.learning_rate * settings.learning_rate_decay^(stage / settings.learning_rate_decay_stages)
        # Keep the original learning-rate type when updating Optimisers.
        rate = convert(typeof(settings.learning_rate), rate)
        Flux.adjust!(optimizer_state, rate)

        total = 0.0
        # Shuffle the sample indices once per epoch
        sample_count = size(training.input, 3)
        shuffled_indices = randperm(rng, sample_count)

        for first_index in 1:settings.batch_size:sample_count
            last_index = min(first_index + settings.batch_size - 1, sample_count)
            indices = shuffled_indices[first_index:last_index]
            current_batch_size = length(indices)

            # Add noise to inputs and keep the original targets.
            input = perturb_state_input(
                training.input[:, :, indices], settings.noise_std, rng
            )
            target = training.target[:, :, indices]

            loss, gradient = Flux.withgradient(model) do current
                predictions = current(input)
                endpoint_loss(predictions, target, 1)
            end
            if !isfinite(loss)
                error("Nonfinite pretraining loss")
            end
            Flux.update!(optimizer_state, model, gradient[1])

            # Weight the loss by the actual batch size.
            total += current_batch_size * loss
        end
        loss = one_step_loss(model, validation, settings.batch_size)
        isfinite(loss) || error("Nonfinite validation loss")
        best_parameters, best_loss, best_epoch = update_best_checkpoint(model, loss, epoch,
            best_parameters, best_loss, best_epoch)
        push!(history, (epoch=epoch, learning_rate=rate,
            training_loss=total / size(training.input, 3), validation_loss=loss))
        if epoch % settings.checkpoint_interval == 0 || epoch == settings.epochs
            checkpoint((; model, optimizer_state, rng, history, best_parameters, best_loss, best_epoch, epoch))
        end
    end
    return (; model, optimizer_state, rng, history, best_parameters, best_loss, best_epoch, epoch=last(history).epoch)
end
