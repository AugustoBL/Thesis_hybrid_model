# Optional: reproduce the flux-correction source pretraining in Appendix A.1.
# The target experiments normally use Trained_models/flux_correction_pretrained.bson.

# 1. Source data, partitions, architecture and optimisation ------------------
# Resolve paths from this script's location, independently of the working directory.
const repository = normpath(joinpath(@__DIR__, ".."))
# This ensemble contains the complete 24-hour source trajectories.
const source_file = joinpath(repository, "Data", "24_hour_source.jld2")
# Keep newly trained models separate from the original thesis checkpoint.
const output_directory = joinpath(repository, "results", "thesis_reproduction", "source_pretraining")
# Event numbers identify independent trajectories within the source ensemble.
const source_training_events = collect(1:4)
# These events select the checkpoint; they never supply optimiser updates.
const source_validation_events = collect(29:34)
# These held-out events measure accuracy after checkpoint selection.
const source_test_events = collect(35:40)
# Each hidden convolution produces 16 feature channels using a three-point kernel.
const hidden_channels = 16
const filter_width = 3
const pretraining_settings = (
    # Train for 500 passes through the adjacent-state pairs, in batches of 50.
    epochs=500, batch_size=50, learning_rate=1e-4,
    # The shared trainer changes the rate every 250 epochs using this decay formula.
    learning_rate_decay=0.01, learning_rate_steps=250, learning_rate_decay_stages=3,
    # h is water level [m]; u is velocity [m/s]. Perturb each input at scale 0.001.
    # Save resume state every 25 epochs.
    # The seed fixes input-noise and training-pair sampling in the shared trainer.
    noise_std=0.001f0, checkpoint_interval=25, seed=2026,
)

# Load the numerical functions only when execution is explicitly requested.
if "--run" in ARGS
    include(joinpath(@__DIR__, "experiment_io.jl"))
end

# Run data preparation, source training and held-out evaluation in that order.
function main()
    # 2. Form every adjacent state pair in the specified source events --------
    # Return the states/grid/physical parameters and the stored-state interval in seconds.
    source_dataset, time_step = ConservativePilot.load_dataset(source_file)
    # Convert each consecutive state pair into five input channels and two target channels.
    # Inputs are padded h, u, depth/30, wind stress and boundary flags; targets are next-step h/u.
    # Build the three partitions separately so training never sees validation or test targets.
    training_pairs = ConservativePilot.one_step_pairs(source_dataset, source_training_events)
    validation_pairs = ConservativePilot.one_step_pairs(source_dataset, source_validation_events)
    test_pairs = ConservativePilot.one_step_pairs(source_dataset, source_test_events)

    # Preserve the notebook's specification fields and hash-derived directory.
    # A SHA-256 digest identifies the exact input-file bytes used for this run.
    source_checksum = bytes2hex(sha256(read(source_file)))
    # Record the implementations that define the architecture and training procedure.
    code_paths = ["src/models.jl", "src/training_utils.jl",
        "thesis_experiments/shared/conservative_pilot_utils.jl"]
    # Store one digest per implementation file, in the same order as code_paths.
    code_checksums = [bytes2hex(sha256(read(joinpath(repository, path)))) for path in code_paths]
    # Bundle event memberships, hyperparameters and physical grid scales for later inspection.
    specification = (source_file=source_file, source_checksum=source_checksum,
        train=source_training_events, validation=source_validation_events, test=source_test_events,
        settings=pretraining_settings, n_hidden=hidden_channels, n_filter=filter_width,
        dt=time_step, dx=source_dataset.params.dx, code_checksums=code_checksums)
    # Use the first 12 digest characters to name this particular pretraining specification.
    pretraining_id = bytes2hex(sha256(repr(specification)))[1:12]
    # The selected checkpoint, final model and training history share this directory.
    directory = joinpath(output_directory, "pretraining", pretraining_id, "flux_correction")

    # 3. Train only the architecture used in the final thesis -----------------
    # Fix the random convolutional weights before constructing a fresh model.
    Random.seed!(pretraining_settings.seed)
    # Build the flux-correction CNN with the source time step dt and cell spacing dx.
    model = Surge.build_flux_correction_cnn(hidden_channels, filter_width;
        dt=time_step, dx=source_dataset.params.dx)
    # Fit one-step state predictions, validate each epoch and retain the lowest validation MSE.
    # record contains that selected model, its training history and its checkpoint path.
    record = ConservativePilot.pretrain(model, training_pairs, validation_pairs, directory;
        settings=pretraining_settings)

    # 4. Save the original selected-model test score and provenance -----------
    # Average the selected model's squared h/u errors over the held-out adjacent pairs.
    # The batch-size argument limits how many pairs are evaluated together.
    test_loss = ConservativePilot.one_step_loss(record.model, test_pairs, pretraining_settings.batch_size)
    # Keep the original JLD2 keys so existing readers can recover provenance and test loss.
    JLD2.jldsave(joinpath(directory, "specification.jld2"); specification=specification, test_loss=test_loss)
    # Make one human-readable row for the reported architecture and selected epoch.
    # trainable_parameter_count counts scalar parameters rather than parameter arrays.
    summary = [(architecture=:flux_correction, best_epoch=record.result.best_epoch,
        validation_mse=record.result.best_validation_loss, source_test_mse=test_loss,
        trainable_parameters=Surge.trainable_parameter_count(record.model), checkpoint=record.path)]
    # Write the summary as a tab-separated table and display the reusable model path.
    Surge.save_table(joinpath(output_directory, "pretraining", pretraining_id, "summary.tsv"), summary)
    println("Selected source checkpoint: $(record.path)")
end

# Execute the workflow with --run; otherwise only describe the prepared experiment.
if "--run" in ARGS
    main()
else
    println("Optional source pretraining: flux-correction CNN, 500 epochs, seed 2026.")
    println("Train events 1-4; validation 29-34; test 35-40. Add --run to execute.")
    println("Output: $output_directory")
end
