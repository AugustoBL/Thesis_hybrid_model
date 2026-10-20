using Test, Flux, JLD2, BSON, TOML
root = isempty(ARGS) ? normpath(joinpath(@__DIR__, "..")) : ARGS[1]
empty!(ARGS)
@testset "Julia syntax and plan-only runners" begin
    files = String[]
    for (directory, _, names) in walkdir(root), name in names
        endswith(name, ".jl") && push!(files, joinpath(directory, name))
    end
    for path in files
        parsed = Meta.parseall(read(path, String))
        function check_parse(x)
            x isa Expr || return true
            x.head in (:error, :incomplete) && return false
            return all(check_parse, x.args)
        end
        @test check_parse(parsed)
    end
    for name in ["00_source_pretraining.jl", "01_main_experiments.jl", "02_gauge_and_curriculum_followups.jl", "03_no_domain_shift_control.jl", "04_profile_adaptation.jl"]
        sandbox = Module(gensym(:Runner))
        Base.include(sandbox, joinpath(root, "thesis_experiments", name))
        @test isdefined(sandbox, :main)
    end
end
include(joinpath(root,"thesis_experiments","experiment_io.jl"))
include(joinpath(root,"thesis_experiments","shared","reviewer_followup_runtime.jl"))
include(joinpath(root,"lora_rank_optimisation","rank_selection.jl"))
@testset "Restored helpers, input files, and model execution" begin
    @test isdefined(ReviewerFollowupRuntime,:fit)
    @test isdefined(ReviewerFollowupRuntime,:evaluate_models)
    model=Surge.load_model(joinpath(root,"Trained_models","flux_correction_pretrained.bson"))
    @test Surge.trainable_parameter_count(model)==2706
    for name in ["uniform","interior","boundary","control"]
        dataset, dt=ConservativePilot.load_dataset(joinpath(root,"Data","thesis_v4",name*".jld2"))
        @test length(dataset.solution)==18
        @test dt>0 && last(dataset.times)>=72*3600-dt
        @test length(dataset.grid.h)==100 && length(dataset.grid.u)==101
        event=dataset.solution[1][1:3]
        forecast=Surge.forecast_event(Surge.effective_model(model),event,dataset.params)
        @test all(isfinite,forecast.prediction)
        @test size(forecast.prediction)==size(forecast.target)
        if name=="uniform"
            input=Surge._model_input(first(event),dataset.params)
            target=Surge._model_target(event[2])
            x=reshape(input,size(input)...,1)
            y=reshape(target,size(target)...,1)
            for method in [:full,:first_two,:last_two,:output,:film,:lora]
                ranks = method == :lora ? TOML.parsefile(joinpath(root,"results","thesis","thesis_flux_correction_v4","rank_selection","selected_ranks.toml"))["folds"]["1"] : nothing
                adapted=Surge.configure_adaptation(model,method;seed=1,ranks=ranks)
                loss,grads=Flux.withgradient(adapted) do m
                    sum(abs2,m(x).-y)/length(y)
                end
                @test isfinite(loss)
                @test grads[1]!==nothing
            end
        end
    end
    source,dt=ConservativePilot.load_dataset(joinpath(root,"Data","24_hour_source.jld2"))
    @test length(source.solution)>=40
    @test last(source.times)>=24*3600-dt
end
@testset "Saved main fits and fold selections" begin
    campaign=joinpath(root,"results","thesis","thesis_flux_correction_v4")
    for fold in 1:5
        @test isfile(joinpath(campaign,"selections","method_fold_$fold.tsv"))
        @test isfile(joinpath(campaign,"selections","temporal_fold_$fold.tsv"))
    end
    ranks=TOML.parsefile(joinpath(campaign,"rank_selection","selected_ranks.toml"))
    @test haskey(ranks,"folds")
    for (directory,_,names) in walkdir(joinpath(campaign,"runs"))
        if "result.jld2" in names
            metadata=JLD2.load(joinpath(directory,"result.jld2"),"result")
            @test hasproperty(metadata,:configuration)
            @test hasproperty(metadata,:split)
            @test isfile(joinpath(directory,"best_model.bson"))
            selected=Surge.load_model(joinpath(directory,"best_model.bson"))
            @test Surge.trainable_parameter_count(selected)>0
        end
    end
end
println("MIGRATION_RUNTIME_VERIFICATION_OK")
