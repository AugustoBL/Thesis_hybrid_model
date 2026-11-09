using Test, Flux, Random, Statistics
include(joinpath(@__DIR__, "..", "src", "Surge.jl"))
using .Surge

@testset "Model adaptation and conservation" begin
    Random.seed!(42)
    base = build_flux_correction_cnn(3, 3; dt=0.01f0, dx=1f0)
    input = 0.01f0 .* randn(Float32, 6, 5, 3)
    input[:, 3, :] .= 1
    input[end, 1, :] .= 0
    reference = base(input)
    @test size(reference) == (6, 2, 3)
    @test sum(reference[1:end-1, 1, :]; dims=1) ≈ sum(input[1:end-1, 1, :]; dims=1)
    @test all(iszero, reference[[1, end], 2, :])
    for method in (:full, :output, :first_two, :last_two, :film, :lora)
        adapted = configure_adaptation(base, method; seed=1, ranks=fill(1, 5))
        @test adapted(input) ≈ reference
        @test Surge.effective_model(adapted)(input) ≈ adapted(input)
        original = deepcopy(Flux.trainables(base))
        Flux.trainables(adapted)[1] .+= 0.1f0
        @test Flux.trainables(base) == original
    end
    lora = configure_adaptation(base, :lora; seed=1, ranks=fill(1, 5))
    gradient = Flux.gradient(m -> sum(abs2, Surge.effective_model(m)(input)), lora)[1]
    @test all(g -> all(isfinite, g), Flux.trainables(gradient))
    @test any(g -> any(!iszero, g), Flux.trainables(gradient))
    rank_zero = configure_adaptation(base, :lora; seed=1, ranks=zeros(Int, 5))
    @test rank_zero(input) ≈ reference
    @test_throws DimensionMismatch Surge.restore_trainable_parameters!(base, [])
    perturbed = Surge.perturb_state_input(input, 0.01f0, MersenneTwister(2))
    @test sum(perturbed[1:end-1, 1, :]; dims=1) ≈ sum(input[1:end-1, 1, :]; dims=1) atol=1f-7
    @test perturbed[[1, end], 2, :] == input[[1, end], 2, :]
    @test perturbed[end, 1, :] == input[end, 1, :]
end

@testset "Window preparation and training" begin
    model = build_base_cnn(5, 2, 2, 3, 2)
    input = 0.01f0 .* randn(Float32, 6, 5, 2)
    forcing = zeros(Float32, 6, 3, 4, 2)
    target = zeros(Float32, 6, 2, 4, 2)
    data = SparseStationEventDataset(input, forcing, target, ones(Float32, size(target)), 1)
    settings = (epochs=1, batch_size=3, rollout_horizons=[1, 2], window_stride=1,
        normalize_loss=true, detach_gradients=false, gradient_steps=1,
        curriculum_enabled=false, loss_mode=:balanced, endpoint_horizon=2,
        validation_interval=1, patience=0)
    windows = Surge.enumerate_training_windows(data, settings)
    @test windows == [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]
    for stations in ([1, 2], (h=[1, 2], u=[2, 3]))
        scales = Surge.compute_pretrained_error_scales(model, data, settings, stations)
        @test length(scales) == 2
        batches = Surge.prepare_window_batches(data, windows, settings, stations)
        @test size(batches[1].forcing) == (6, 3, 2, 3)
        states = Surge.precompute_rollout_states(model, data)
        batch = Surge.prepare_objective_batch!(batches[1], states, settings, [2], scales)
        @test batch.state[:, :, 1] == states[:, :, 1, 1]
        @test batch.horizons == [2]
        update_model = deepcopy(model)
        loss = Surge.optimizer_step!(update_model, Flux.setup(Adam(1f-4), update_model), batch, stations)
        @test loss isa NamedTuple
        @test isfinite(loss.loss)
    end
    checkpoints = []
    result = Surge.train_model!(model, data, data; settings, stations=[1, 2],
        normalizers=ones(2), optimizer=Adam(1f-4), rng=MersenneTwister(1),
        checkpoint=state -> push!(checkpoints, state))
    @test result.updates == 2
    @test length(result.history) == 2
    @test propertynames(result.history[1]) == propertynames(result.history[2])
    @test length(checkpoints) == 1
    @test Surge.validation_loss(model, data, (h=[1, 2], u=[2])) == Surge.validation_loss(model, data, [1, 2])
    @test_throws ArgumentError Surge.compute_pretrained_error_scales(model, data,
        merge(settings, (rollout_horizons=[8],)), [1, 2])
    pairs = (input=randn(Float32, 6, 5, 5), target=zeros(Float32, 6, 2, 5))
    @test Surge.one_step_loss(model, pairs, 3) ≈ Surge.endpoint_loss(model(pairs.input), pairs.target, 1)
    pretrain_settings = (epochs=1, batch_size=3, learning_rate=1f-4, seed=1,
        learning_rate_steps=2, learning_rate_decay_stages=3, learning_rate_decay=0.1,
        noise_std=0f0, checkpoint_interval=1)
    pretrained = Surge.train_model!(model, pairs, pairs; settings=pretrain_settings)
    @test pretrained.epoch == 1
    @test all(isfinite, [h.validation_loss for h in pretrained.history])
    constant_rate = Surge.train_model!(model, pairs, pairs;
        settings=merge(pretrain_settings, (learning_rate_decay_stages=0,)))
    @test constant_rate.history[end].learning_rate == pretrain_settings.learning_rate
end
