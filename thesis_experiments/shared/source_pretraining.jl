module SourcePretraining

using ..Surge, JLD2

"""Persist source pretraining using the shared supervised trainer."""
function pretrain(model, training, validation, directory; specification)
    settings = specification.settings
    mkpath(directory)
    result_path = joinpath(directory, "complete.jld2")
    best_path = joinpath(directory, "best_model.bson")
    if isfile(result_path)
        result = JLD2.load(result_path, "result")
        isequal(result.specification, specification) || error("Saved pretraining settings differ: $directory")
        return (model=Surge.load_model(best_path), result=result, path=best_path)
    end
    checkpoint_path = joinpath(directory, "checkpoint.jld2")
    saved = isfile(checkpoint_path) ? JLD2.load(checkpoint_path, "saved") : nothing
    if saved !== nothing
        isequal(saved.specification, specification) || error("Checkpoint settings differ: $directory")
    end
    resume = saved === nothing ? nothing : saved.state
    checkpoint = function(state)
        JLD2.jldsave(checkpoint_path * ".tmp"; saved=(; state, specification))
        mv(checkpoint_path * ".tmp", checkpoint_path; force=true)
        Surge.save_table(joinpath(directory, "history.tsv"), state.history)
        println("Pretraining epoch $(state.epoch)/$(settings.epochs), validation MSE = $(state.best_loss)")
        flush(stdout)
    end
    state = Surge.train_model!(model, training, validation; settings, resume, checkpoint)
    Surge.restore_trainable_parameters!(state.model, state.best_parameters)
    Surge.save_model(best_path, state.model)
    result = (history=state.history, best_epoch=state.best_epoch,
        best_validation_loss=state.best_loss, settings=settings, specification=specification)
    JLD2.jldsave(result_path * ".tmp"; result)
    mv(result_path * ".tmp", result_path; force=true)
    return (model=state.model, result=result, path=best_path)
end

end
