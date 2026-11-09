module SourcePretraining

using ..Surge, JLD2

function pretrain(model, training, validation, directory; specification)
    settings = specification.settings
    mkpath(directory)
    result_path = joinpath(directory, "complete.jld2")
    best_path = joinpath(directory, "best_model.bson")
    if isfile(result_path)
        # Reuse the completed fit only if the data, model and settings match.
        result = JLD2.load(result_path, "result")
        isequal(result.specification, specification) || error("Saved pretraining settings differ: $directory")
        return (model=Surge.load_model(best_path), result=result, path=best_path)
    end
    checkpoint_path = joinpath(directory, "checkpoint.jld2")
    # Save training progress and loss history.
    checkpoint = function(state)
        JLD2.jldsave(checkpoint_path * ".tmp"; saved=(; state, specification))
        mv(checkpoint_path * ".tmp", checkpoint_path; force=true)
        Surge.save_table(joinpath(directory, "history.tsv"), state.history)
        println("Pretraining epoch $(state.epoch)/$(settings.epochs), validation MSE = $(state.best_loss)")
        flush(stdout)
    end
    # Train from the start and restore the best validation parameters.
    state = Surge.train_model!(model, training, validation; settings, checkpoint)
    Surge.restore_trainable_parameters!(state.model, state.best_parameters)
    Surge.save_model(best_path, state.model)
    # Save the completed fit and its settings for later reuse.
    result = (history=state.history, best_epoch=state.best_epoch,
        best_validation_loss=state.best_loss, settings=settings, specification=specification)
    JLD2.jldsave(result_path * ".tmp"; result)
    mv(result_path * ".tmp", result_path; force=true)
    return (model=state.model, result=result, path=best_path)
end

end
