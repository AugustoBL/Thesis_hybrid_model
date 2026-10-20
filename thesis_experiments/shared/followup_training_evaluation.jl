module ReviewerFollowupRuntime

using ..Surge, ..ExperimentDiagnostics
using Flux, JLD2, Random, Statistics

"""Train with the shared thesis loop; completion and validation checkpoints support resume."""
function fit(base, dataset, split, cfg, directory; evaluate_only=false)
    result_path = joinpath(directory,"result.jld2")
    if isfile(result_path)
        result = JLD2.load(result_path,"result")
        isequal(result.configuration,cfg) && isequal(result.split,split) ||
            error("Saved run settings differ: $directory")
        isfile(joinpath(directory,"best_model.bson")) || error("Missing selected model: $directory")
        return result
    end
    evaluate_only && error("Incomplete training run: $directory")
    mkpath(directory)
    checkpoint_path = joinpath(directory,"checkpoint.jld2")
    previous = isfile(checkpoint_path) ? JLD2.load(checkpoint_path,"saved") : nothing
    if previous !== nothing
        isequal(previous.configuration,cfg) && isequal(previous.split,split) ||
            error("Checkpoint settings differ: $directory")
    end
    start = time_ns()
    training, validation = Surge.prepare_training_split(dataset,split,cfg)
    norms = Surge.compute_pretrained_error_scales(base,training,cfg.training,cfg.stations)
    all(isfinite,norms) || error("Nonfinite pretrained normalization scales")
    preparation_seconds = (time_ns()-start)/1e9 + (previous === nothing ? 0.0 : previous.preparation_seconds)
    model = Surge.configure_adaptation(base,:full;seed=split.seed)
    prior = previous === nothing ? 0.0 : previous.training_seconds
    start = time_ns()
    checkpoint = function(state)
        saved = (state=state,configuration=cfg,split=split,preparation_seconds=preparation_seconds,
            training_seconds=prior+(time_ns()-start)/1e9)
        JLD2.jldsave(checkpoint_path*".tmp";saved)
        mv(checkpoint_path*".tmp",checkpoint_path;force=true)
        Surge.save_table(joinpath(directory,"training_history.tsv"),state.history)
        println("  epoch $(state.epoch)/$(cfg.training.epochs), validation=$(state.best_loss)")
        flush(stdout)
    end
    state = Surge.train_model!(model,training,validation;settings=cfg.training,stations=cfg.stations,
        normalizers=norms,optimizer=Adam(cfg.optimizer.learning_rate,cfg.optimizer.betas,cfg.optimizer.epsilon),
        rng=MersenneTwister(split.seed+cfg.optimizer_seed_offset),checkpoint=checkpoint,
        resume=previous === nothing ? nothing : previous.state)
    training_seconds = prior+(time_ns()-start)/1e9
    Surge.save_model(joinpath(directory,"final_model.bson"),state.model)
    Surge.restore_trainable_parameters!(state.model,state.best_parameters)
    Surge.save_model(joinpath(directory,"best_model.bson"),state.model)
    result = (configuration=cfg,split=split,history=state.history,best_epoch=state.best_epoch,
        best_validation_loss=state.best_loss,completed_epochs=state.epoch,updates=state.updates,
        trainable_parameters=Surge.trainable_parameter_count(state.model),normalizers=norms,
        training_seconds=training_seconds,preparation_seconds=preparation_seconds)
    JLD2.jldsave(result_path*".tmp";result)
    mv(result_path*".tmp",result_path;force=true)
    return result
end

"""Evaluate every model with identical inputs and untouched future reference states.

The perturbed case additionally includes the clean-trained adapted baseline
under the same initial-state error, separating input sensitivity from retraining.
Only interval metrics are retained to avoid duplicating large forecast archives.
"""
function evaluate_models(models, dataset, split, stations; forecast_hours=72.0,
                         interval_boundaries=[0.0,12.0,24.0,48.0,72.0])
    count = searchsortedlast(dataset.times,forecast_hours*3600)
    count > 0 || error("Empty forecast interval")
    times = dataset.times[1:count]
    n = length(dataset.grid.h)
    interior = n == 100 ? collect(9:92) : collect(1:n)
    rows = NamedTuple[]
    for (label, model) in models, index in split.test
        event = dataset.solution[index][1:count+1]
        forecast = Surge.forecast_event(Surge.effective_model(model),event,dataset.params)
        finite = all(isfinite,forecast.prediction)
        series = ExperimentDiagnostics.error_series(forecast.prediction,forecast.target,first(event).h,
            dataset.params,stations,interior)
        for summary in ExperimentDiagnostics.summarize_error_series(series,times,interval_boundaries)
            push!(rows,(model=label,event=index,finite_rollout=finite,summary...))
        end
    end
    return rows
end

"""Validate relocated/repacked data against all saved frozen error trajectories.

This does not assert byte identity: the new campaign records both hashes and
the numerical validation. Old manifests are never changed. Any discrepancy
beyond Float32-scale tolerances stops the campaign before fitting.
"""
function verify_reference_series(base,dataset,references,splits;rtol=1e-5,atol=1e-7,mean_bias_atol=1e-6)
    fields=(:mse_h,:mse_u,:anomaly_mse,:mean_bias)
    maximum_errors=Dict(k=>0.0 for k in fields)
    checked=Int[]
    model=Surge.effective_model(base)
    for split in filter(s->s.repeat==1,splits)
        dir,_=references[(:gauges_two,split.fold,split.repeat)]
        path=joinpath(dir,"diagnostics","report_metrics.jld2")
        isfile(path) || error("Cannot verify changed data bytes without historical diagnostics: $path")
        report=JLD2.load(path,"result")
        count=length(report.times)
        count==searchsortedlast(dataset.times,72*3600) || error("Historical forecast duration differs")
        all(isapprox(a,b;rtol=1e-8,atol=1e-6) for (a,b) in zip(report.times,dataset.times[1:count])) ||
            error("Historical forecast timestamps differ")
        for index in split.test
            old=only(filter(r->r.model==:pretrained && r.event==index,report.series)).series
            event=dataset.solution[index][1:count+1]
            forecast=Surge.forecast_event(model,event,dataset.params)
            rows=ExperimentDiagnostics.error_series(forecast.prediction,forecast.target,
                first(event).h,dataset.params,[2,4,97,99],9:92)
            length(rows)==length(old) || error("Historical event length differs: $index")
            for k in fields, (a,b) in zip(rows,old)
                x,y=getproperty(a,k),getproperty(b,k)
                tolerance=k==:mean_bias ? mean_bias_atol : atol
                isfinite(x) && isfinite(y) && isapprox(x,y;rtol,atol=tolerance) ||
                    error("Dataset compatibility failed: event $index, step $(a.step), $k: $x versus $y")
                maximum_errors[k]=max(maximum_errors[k],abs(x-y))
            end
            push!(checked,index)
            println("  historical frozen diagnostics agree: event $index")
            flush(stdout)
        end
    end
    sort(checked)==collect(1:18) || error("Compatibility check must cover all 18 events")
    return (mode=:diagnostic_reproduction,events=sort(checked),rtol=rtol,atol=atol,mean_bias_atol=mean_bias_atol,
        maximum_errors=[(field=k,maximum_absolute_error=maximum_errors[k]) for k in fields])
end

"""Average squared errors across repeats within events, then weight events equally."""
function aggregate(rows)
    event_groups = Dict{Any,Vector{NamedTuple}}()
    for r in rows
        key = (r.case,r.model,r.event,r.from_hours,r.to_hours)
        push!(get!(event_groups,key,NamedTuple[]),r)
    end
    events = NamedTuple[]
    for (key, values) in event_groups
        case, model, event, lower, upper = key
        push!(events,(case=case,model=model,event=event,from_hours=lower,to_hours=upper,
            repeats=length(values),finite_rollout=all(r.finite_rollout for r in values),
            mse_h=mean(r.mse_h for r in values),mse_u=mean(r.mse_u for r in values),
            anomaly_mse=mean(r.anomaly_mse for r in values),bias_squared=mean(r.bias_rmse^2 for r in values)))
    end
    sort!(events;by=r->(string(r.case),string(r.model),r.event,r.from_hours))
    groups = Dict{Any,Vector{NamedTuple}}()
    for r in events
        key = (r.case,r.model,r.from_hours,r.to_hours)
        push!(get!(groups,key,NamedTuple[]),r)
    end
    summaries = NamedTuple[]
    for (key, values) in groups
        case,model,lower,upper = key
        push!(summaries,(case=case,model=model,from_hours=lower,to_hours=upper,events=length(values),
            minimum_repeats=minimum(r.repeats for r in values),
            finite_rollout=all(r.finite_rollout for r in values),
            rmse_h=sqrt(mean(r.mse_h for r in values)),rmse_u=sqrt(mean(r.mse_u for r in values)),
            anomaly_rmse=sqrt(mean(r.anomaly_mse for r in values)),
            bias_rmse=sqrt(mean(r.bias_squared for r in values))))
    end
    sort!(summaries;by=r->(string(r.case),string(r.model),r.from_hours))
    return events,summaries
end

"""Paired event comparisons; never silently compare different test memberships."""
function comparisons(events)
    specifications = [(:gauge_one,:adapted,:gauges_two,:adapted),
        (:gauge_one,:adapted,:gauges_four,:adapted),
        (:initial_state,:adapted,:initial_state,:pretrained),
        (:initial_state,:adapted,:initial_state,:clean_adapted),
        (:detached64,:adapted,:long128,:adapted),
        (:detached64,:adapted,:detached16,:adapted),
        (:horizon256,:adapted,:multi128,:adapted),
        (:curriculum_fast,:adapted,:multi128,:adapted),
        (:curriculum_fast,:adapted,:curriculum_original,:adapted)]
    rows = NamedTuple[]
    intervals = sort(unique((r.from_hours,r.to_hours) for r in events))
    for (case,model,reference,reference_model) in specifications, (lower,upper) in intervals
        selected = filter(r->r.case==case && r.model==model && r.from_hours==lower && r.to_hours==upper,events)
        baseline = filter(r->r.case==reference && r.model==reference_model && r.from_hours==lower && r.to_hours==upper,events)
        if isempty(selected) || isempty(baseline)
            continue
        end
        sort!(selected;by=r->r.event); sort!(baseline;by=r->r.event)
        [r.event for r in selected]==[r.event for r in baseline] || error("Unpaired test events in $case versus $reference")
        [r.repeats for r in selected]==[r.repeats for r in baseline] || error("Unpaired repeats in $case versus $reference")
        a=sqrt(mean(r.mse_h for r in selected)); b=sqrt(mean(r.mse_h for r in baseline))
        push!(rows,(case=case,model=model,reference=reference,reference_model=reference_model,
            from_hours=lower,to_hours=upper,events=length(selected),
            rmse_h=a,reference_rmse_h=b,change_percent=b==0 ? NaN : 100*(a/b-1),
            events_improved=count(a.mse_h < b.mse_h for (a,b) in zip(selected,baseline)),
            finite_rollout=all(r.finite_rollout for r in vcat(selected,baseline))))
    end
    return rows
end

end
