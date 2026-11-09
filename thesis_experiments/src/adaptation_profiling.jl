module AdaptationProfiling

using ..Surge
using Flux, Random, Statistics

function build_actions(models, methods, objective, training_data, validation_data, stations)
    # Time each operation separately and record memory allocations and garbage collection.
    actions = Dict{Symbol, Any}()
    for method in methods
        model = models[method]
        # Reuse a fixed gradient and update a model copy when timing the optimizer.
        gradient = Flux.gradient(objective, model)[1]
        update_model = deepcopy(model)
        optimizer = Flux.setup(Flux.Adam(3e-6), update_model)
        actions[method] = (
            kernel_materialization=() -> Surge.effective_model(model),
            forward=() -> objective(model),
            forward_reverse=() -> Flux.gradient(objective, model),
            optimizer=() -> Flux.update!(optimizer, update_model, gradient),
            state_regeneration=() -> Surge.precompute_rollout_states(model, training_data),
            validation=() -> Surge.validation_loss(model, validation_data, stations))
    end

    return actions
end

function measure_actions(actions, methods; fold, samples, warmup)
    # Warm up each method and operation before timing to exclude initial compilation.
    phases = propertynames(actions[first(methods)])
    println("Warming all six methods; compilation is excluded from measured samples.")
    for method in methods
        for phase in phases
            for warmup_index in 1:warmup
                getproperty(actions[method], phase)()
            end
        end
    end

    rows = NamedTuple[]
    random_generator = MersenneTwister(20261016)
    # Shuffle methods and operations for each timing sample to reduce order effects.
    for sample in 1:samples
        for method in shuffle(random_generator, methods)
            for phase in shuffle(random_generator, collect(phases))
                # Run garbage collection before timing. Record collection during the operation.
                GC.gc()
                measurement = @timed getproperty(actions[method], phase)()
                push!(rows, (method=string(method), fold=fold, sample=sample, phase=string(phase),
                    seconds=measurement.time, allocated_bytes=measurement.bytes, gc_seconds=measurement.gctime))
            end
        end
        println("Completed timing sample $sample/$samples")
    end
    return rows
end

function summarize_measurements(rows, methods, phases)
    summary = NamedTuple[]
    for method in methods
        for phase in phases
            selected = filter(row -> row.method == string(method) && row.phase == string(phase), rows)
            times = [row.seconds for row in selected]
            push!(summary, (method=string(method), phase=string(phase), median_seconds=median(times),
                q25_seconds=quantile(times, 0.25), q75_seconds=quantile(times, 0.75),
                median_allocated_bytes=median([row.allocated_bytes for row in selected])))
        end
    end
    return summary
end

end
