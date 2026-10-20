module ConservativePilot

using ..Surge, Flux, JLD2, Random, Statistics

"""Load saved snapshots with forecast times measured from the initial state."""
function load_dataset(path)
    raw = JLD2.load(path)
    times = Float64.(raw["times"])
    intervals = diff(times)
    @assert !isempty(intervals) && first(intervals) > 0
    @assert all(t -> isapprox(t, first(intervals); rtol=1e-6), intervals) "Uniform timestamps required"
    @assert all(length(event) == length(times) for event in raw["solution"])
    dataset = (solution=raw["solution"], grid=raw["grid"], params=raw["params"],
        times=times[2:end] .- first(times))
    return dataset, first(intervals)
end

function steps_until(dataset, hours; require_coverage=true)
    dt = first(dataset.times)
    if require_coverage
        @assert last(dataset.times) >= hours * 3600 - dt "Dataset is shorter than the requested duration"
    end
    count = searchsortedlast(dataset.times, hours * 3600 + 1e-6)
    @assert count > 0
    return count
end

"""Every adjacent pair, from selected events only; no temporal subsampling."""
function one_step_pairs(dataset, indices)
    @assert !isempty(indices) && all(i -> 1 <= i <= length(dataset.solution), indices)
    count = sum(length(dataset.solution[i]) - 1 for i in indices)
    faces = length(dataset.grid.u)
    inputs = Array{Float32}(undef, faces, 5, count)
    targets = Array{Float32}(undef, faces, 2, count)
    sample = 0
    for event_index in indices
        event = dataset.solution[event_index]
        for step in 1:length(event)-1
            sample += 1
            inputs[:, :, sample] .= Surge._model_input(event[step], dataset.params)
            targets[:, :, sample] .= Surge._model_target(event[step+1])
        end
    end
    return (input=inputs, target=targets)
end

# Supervise the entire recurrent state, including the zero-target padded h row.
# ResidualBlock predicts and feeds that row back; leaving it out of the loss
# creates an unconstrained feedback path. Conservative models already zero it.
function one_step_mse(prediction, target)
    return sum(abs2, prediction .- target) / length(target)
end

function one_step_loss(model, data, batch_size)
    total = 0.0
    for indices in Iterators.partition(axes(data.input, 3), batch_size)
        total += length(indices) * one_step_mse(model(data.input[:, :, indices]), data.target[:, :, indices])
    end
    return total / size(data.input, 3)
end

function noisy_input(input, level, rng)
    level == 0 && return input
    perturbed = copy(input)
    noise_h = level .* randn(rng, Float32, size(input, 1)-1, 1, size(input, 3))
    noise_h .-= mean(noise_h; dims=1) # Do not change the volume the target must conserve.
    perturbed[1:end-1, 1:1, :] .+= noise_h
    perturbed[2:end-1, 2:2, :] .+= level .* randn(rng, Float32, size(input, 1)-2, 1, size(input, 3))
    return perturbed
end

"""Teacher-forced one-step pretraining; all epochs, with resumable checkpoints."""
function pretrain(model, training, validation, directory; settings)
    @assert settings.learning_rate_steps > 0 && settings.learning_rate_decay_stages > 0
    @assert 0 < settings.learning_rate_decay <= 1
    mkpath(directory)
    final_path = joinpath(directory, "complete.jld2")
    # The shared selected checkpoint was promoted out of the pilot result folder.
    shared_pilot = normpath(joinpath(@__DIR__, "..", "..", "results", "reported",
        "source_pretraining", "pretraining", "0ca796b4d228", "flux_correction"))
    best_path = normpath(directory) == shared_pilot ?
        normpath(joinpath(@__DIR__, "..", "..", "Trained_models", "flux_correction_pretrained.bson")) :
        joinpath(directory, "best_model.bson")
    if isfile(final_path)
        return (model=load_model(best_path), result=JLD2.load(final_path, "result"), path=best_path)
    end
    checkpoint_path = joinpath(directory, "checkpoint.jld2")
    if isfile(checkpoint_path)
        saved = JLD2.load(checkpoint_path, "state")
        model, optimizer, rng = saved.model, saved.optimizer, saved.rng
        history, best, best_loss, best_epoch = saved.history, saved.best, saved.best_loss, saved.best_epoch
        first_epoch = saved.epoch + 1
    else
        optimizer = Flux.setup(Adam(settings.learning_rate), model)
        rng = MersenneTwister(settings.seed)
        best_loss = one_step_loss(model, validation, settings.batch_size)
        @assert isfinite(best_loss) "Nonfinite initial one-step validation loss"
        best, best_epoch = Surge.copy_trainable_parameters(model), 0
        history = NamedTuple[(epoch=0, learning_rate=settings.learning_rate,
            training_loss=NaN, validation_loss=best_loss)]
        first_epoch = 1
    end
    for epoch in first_epoch:settings.epochs
        # A shorter run truncates the same schedule instead of accelerating decay.
        stage = min(fld(epoch-1, settings.learning_rate_steps), settings.learning_rate_decay_stages)
        rate = settings.learning_rate * settings.learning_rate_decay^(stage / settings.learning_rate_decay_stages)
        Flux.adjust!(optimizer, rate)
        total = 0.0
        for indices in Iterators.partition(randperm(rng, size(training.input, 3)), settings.batch_size)
            input = noisy_input(training.input[:, :, indices], settings.noise_std, rng)
            target = training.target[:, :, indices]
            loss, gradient = Flux.withgradient(model) do current
                one_step_mse(current(input), target)
            end
            @assert isfinite(loss) "Nonfinite pretraining loss at epoch $epoch"
            @assert all(p -> all(isfinite, p), Flux.trainables(gradient[1])) "Nonfinite pretraining gradient"
            Flux.update!(optimizer, model, gradient[1])
            total += length(indices) * loss
        end
        validation_loss = one_step_loss(model, validation, settings.batch_size)
        if isfinite(validation_loss) && validation_loss < best_loss
            best, best_loss, best_epoch = Surge.copy_trainable_parameters(model), validation_loss, epoch
        end
        push!(history, (epoch=epoch, learning_rate=rate,
            training_loss=total / size(training.input, 3), validation_loss=validation_loss))
        if epoch % settings.checkpoint_interval == 0 || epoch == settings.epochs
            state = (; model, optimizer, rng, history, best, best_loss, best_epoch, epoch)
            JLD2.jldsave(checkpoint_path; state)
            Surge.save_table(joinpath(directory, "history.tsv"), history)
            println("Pretraining epoch $epoch/$(settings.epochs), validation MSE = $validation_loss")
            flush(stdout)
        end
    end
    save_model(joinpath(directory, "final_model.bson"), model)
    Surge.restore_trainable_parameters!(model, best)
    save_model(best_path, model)
    result = (history=history, best_epoch=best_epoch, best_validation_loss=best_loss, settings=settings)
    JLD2.jldsave(final_path; result)
    return (model=model, result=result, path=best_path)
end

"""Free rollout from t=0; target states after initialization never enter the model."""
function evaluate(model, dataset, event_indices; training_hours, forecast_hours)
    steps = steps_until(dataset, forecast_hours)
    times = dataset.times[1:steps]
    cells = length(dataset.grid.h)
    h_error, u_error, drift = zeros(steps), zeros(steps), zeros(steps)
    windows = (("0-12h", findall(t -> t <= training_hours*3600, times)),
        ("12-72h", findall(t -> t > training_hours*3600, times)),
        ("0-72h", collect(eachindex(times))))
    rows = NamedTuple[]
    example = nothing
    model = Surge.effective_model(model)
    for event_index in event_indices
        event = dataset.solution[event_index][1:steps+1]
        forecast = forecast_event(model, event, dataset.params)
        prediction, target = forecast.prediction, forecast.target
        eh = vec(mean(abs2.(Float64.(prediction[1:cells, 1, :]) .- target[1:cells, 1, :]); dims=1))
        eu = vec(mean(abs2.(Float64.(prediction[:, 2, :]) .- target[:, 2, :]); dims=1))
        volume_drift = dataset.params.W * dataset.params.dx .* abs.(
            vec(sum(Float64.(prediction[1:cells, 1, :]); dims=1)) .- sum(Float64.(first(event).h)))
        finite_steps = [all(isfinite, prediction[:, :, step]) for step in 1:steps]
        failure = findfirst(!, finite_steps)
        finite = failure === nothing
        # Keep failed forecasts visibly failed; never average away NaNs or drop events.
        h_error .+= ifelse.(isfinite.(eh), eh, Inf)
        u_error .+= ifelse.(isfinite.(eu), eu, Inf)
        drift = max.(drift, ifelse.(isfinite.(volume_drift), volume_drift, Inf))
        for (label, indices) in windows
            @assert !isempty(indices)
            push!(rows, (event=event_index, window=label,
                mse_h=all(isfinite, eh[indices]) ? mean(eh[indices]) : Inf,
                mse_u=all(isfinite, eu[indices]) ? mean(eu[indices]) : Inf,
                max_volume_drift=all(isfinite, volume_drift[indices]) ? maximum(volume_drift[indices]) : Inf,
                finite_rollout=finite, first_nonfinite_hour=finite ? NaN : times[failure]/3600))
        end
        example === nothing && (example=(event=event_index, prediction=prediction, target=target))
    end
    return (rows=rows, times=times, mse_h=h_error ./ length(event_indices),
        mse_u=u_error ./ length(event_indices), max_volume_drift=drift, example=example)
end

end
