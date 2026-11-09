
"""
Wrap a CNN that predicts increments to the state channels.

Args:
    f: CNN predicting state increments.
    n_state_channels: Number of input channels advanced by the model.

Returns:
    ResidualBlock holding the supplied CNN.
"""
struct ResidualBlock
    f::Chain
    n_state_channels::Int
end

"""
Add predicted increments to the input state.

Args:
    model: ResidualBlock with the increment CNN.
    input: Features `(space, input_channels, batch)`.

Returns:
    Updated state `(space, n_state_channels, batch)`.
"""
function (model::ResidualBlock)(input)
    return input[:, 1:model.n_state_channels, :] + model.f(input)
end

Flux.@layer ResidualBlock

"""
Combine learned velocity increments with water-level updates.

Args:
    f: CNN predicting velocity increments and flux corrections.
    dt: Model timestep [s].
    dx: Cell spacing [m].
    depth_scale: Depth normalization scale [m].
    mode: :velocity_only or :flux_correction (adds a learned transport correction).

Returns:
    ConservativeVelocityModel.
"""
struct ConservativeVelocityModel{C,T}
    f::C
    dt::T
    dx::T
    depth_scale::T
    mode::Symbol
end
Flux.@layer ConservativeVelocityModel trainable=(f,)

"""
Advance water level and velocity with zero boundary flux.

Args:
    model: Conservative wrapper.
    input: Features `(N + 1, 5, batch)`: padded h [m], face u [m/s], depth/depth_scale, wind stress and boundary indicator.

Returns:
    State `(N + 1, 2, batch)` with padded h [m] and face u [m/s].
"""
function (model::ConservativeVelocityModel)(input)
    # The CNN predicts velocity increments and optional transport corrections.
    learned = model.f(input)
    # Discard the padded water-level entry.
    h = input[1:end-1, 1:1, :]
    boundary = zero.(input[1:1, 2:2, :])
    # Update only interior face velocities
    u = cat(boundary, input[2:end-1, 2:2, :] .+ learned[2:end-1, 1:1, :], boundary; dims=1)
    # Average cell water levels at the faces.
    h_face = cat(h[1:1, :, :], (h[1:end-1, :, :] .+ h[2:end, :, :]) ./ 2,
        h[end:end, :, :]; dims=1)
    # Undo the depth scaling and add water levels to get total depth.
    depth = model.depth_scale .* input[:, 3:3, :] .+ h_face
    interior_flux = depth[2:end-1, :, :] .* u[2:end-1, :, :]
    if model.mode == :flux_correction
        # The learned flux correction can compensate for velocity errors.
        interior_flux = interior_flux .+ learned[2:end-1, 2:2, :]
    end
    flux = cat(boundary, interior_flux, boundary; dims=1)
    # Update water levels from the flux difference.
    next_h = h .- (model.dt / model.dx) .* (flux[2:end, :, :] .- flux[1:end-1, :, :])
    # Add the zero h padding row and stack h and u as two channels.
    return cat(cat(next_h, boundary; dims=1), u; dims=2)
end

"""
Access the CNN stored in a model wrapper.

Args:
    model: Wrapper with the f network field.

Returns:
    The stored CNN without copying.
"""
network(model) = model.f

"""
Replace the CNN.

Args:
    model: Original residual wrapper.
    cnn: Replacement CNN.

Returns:
    ResidualBlock with the supplied CNN.
"""
with_network(model::ResidualBlock, cnn) = ResidualBlock(cnn, model.n_state_channels)

"""
Replace the CNN in the wrapper.

Args:
    model: Original conservative wrapper.
    cnn: Replacement CNN.

Returns:
    ConservativeVelocityModel.
"""
with_network(model::ConservativeVelocityModel, cnn) =
    ConservativeVelocityModel(cnn, model.dt, model.dx, model.depth_scale, model.mode)

"""
Build a five-layer residual CNN.

Args:
    n_input_channels: Number of input feature channels.
    n_output_channels: Number of predicted increment channels.
    n_hidden_channels: Number of hidden channels.
    filter_width: Convolution kernel width in grid points.
    n_state_channels: Number of state channels receiving increments (h,u normally).
    activation: Hidden-layer activation function.
    pad: Convolution padding in grid points.

Returns:
    ResidualBlock wrapping the five-layer CNN.
"""
function build_base_cnn(
    n_input_channels::Int,
    n_output_channels::Int,
    n_hidden_channels::Int,
    filter_width::Int,
    n_state_channels::Int;
    activation=swish,
    pad=1,
)

    network = Chain(
        Conv((filter_width,), n_input_channels => n_hidden_channels, activation; pad=pad),
        Conv((filter_width,), n_hidden_channels => n_hidden_channels, activation; pad=pad),
        Conv((filter_width,), n_hidden_channels => n_hidden_channels, activation; pad=pad),
        Conv((filter_width,), n_hidden_channels => n_hidden_channels, activation; pad=pad),
        Conv((filter_width,), n_hidden_channels => n_output_channels; pad=pad),
    )
    return ResidualBlock(network, n_state_channels)
end

"""
Build a conservative CNN with optional flux correction.

Args:
    n_hidden_channels: Number of channels.
    filter_width: Convolution kernel width.
    dt: Model timestep [s].
    dx: Cell spacing [m].
    depth_scale: Depth normalization scale [m] found in the data.
    mode: Either :velocity_only or :flux_correction.
    activation: Hidden-layer activation function.

Returns:
    ConservativeVelocityModel.
"""
function build_conservative_velocity_cnn(n_hidden_channels::Int, filter_width::Int;
    dt, dx, depth_scale=30f0, mode=:velocity_only, activation=swish)
    mode in (:velocity_only, :flux_correction) || throw(ArgumentError("Unsupported conservative mode: $mode"))
    isodd(filter_width) || throw(ArgumentError("filter_width must be odd to preserve spatial length"))
    channels = mode == :velocity_only ? 1 : 2
    cnn = build_base_cnn(5, channels, n_hidden_channels, filter_width, 2;
        activation=activation, pad=filter_width ÷ 2).f
    if mode == :flux_correction
        # Start the added flux at zero
        cnn.layers[end].weight[:, :, 2] .= 0
        cnn.layers[end].bias[2] = 0
    end
    return ConservativeVelocityModel(cnn, Float32(dt), Float32(dx), Float32(depth_scale), mode)
end

"""
Build a conservative CNN with a learned flux correction.

Args:
    n_hidden_channels: Number of hidden channels.
    filter_width: Convolution kernel width in grid points.
    kwargs: Timestep, grid spacing, depth scale and activation forwarded to the conservative builder.

Returns:
    ConservativeVelocityModel in :flux_correction mode.
"""
build_flux_correction_cnn(n_hidden_channels::Int, filter_width::Int; kwargs...) =
    build_conservative_velocity_cnn(n_hidden_channels, filter_width; mode=:flux_correction, kwargs...)

"""
Store convolution parameters excluded from optimizer updates. Used for layer freezing.

Args:
    weight: Convolution kernel.
    bias: Output-channel biases.
    activation: Convolution activation function.
    pad: Convolution padding.

Returns:
    FrozenConv with fixed kernel and bias.
"""
struct FrozenConv
    weight::AbstractArray
    bias::AbstractArray
    activation
    pad
end
Flux.@layer FrozenConv trainable=()

"""
Copy a convolution into a frozen layer.

Args:
    convolution: Base convolution to copy.

Returns:
    FrozenConv.
"""
function FrozenConv(convolution::Conv)
    return FrozenConv(copy(convolution.weight), copy(convolution.bias), convolution.σ, convolution.pad)
end

"""
Apply a frozen convolution.

Args:
    layer: Frozen convolution.
    input: Input tensor `(space, input_channels, batch)`.

Returns:
    Output tensor `(space, output_channels, batch)`.
"""
function (layer::FrozenConv)(input)
    weight, bias = Flux.ignore_derivatives() do
        layer.weight, layer.bias
    end
    output = Flux.conv(input, weight; pad=layer.pad) .+ reshape(bias, 1, :, 1)
    return layer.activation.(output)
end

"""
Store a frozen convolution with trainable per-channel scale and shift.

Args:
    frozen_weight: Fixed kernel `(kernel_width, input_channels, output_channels)`.
    frozen_bias: Fixed vector with one bias per output channel.
    scale: Trainable vector with one value per output channel.
    shift: Trainable vector with one value per output channel.
    activation: Activation function applied after modulation.
    pad: Convolution padding.

Returns:
    FilmConv with output-channel scale and shift.
"""
struct FilmConv
    frozen_weight::AbstractArray
    frozen_bias
    scale::AbstractArray
    shift::AbstractArray
    activation
    pad
end

Flux.@layer FilmConv trainable=(scale, shift)

"""
Wrap a convolution with FiLM modulation.

Args:
    conv: Base convolution.

Returns:
    FilmConv with unit scales and zero shifts. Base arrays are shared with conv.
"""
function FilmConv(conv::Conv)
    n_output_channels = size(conv.weight, 3)
    return FilmConv(
        conv.weight,
        conv.bias,
        ones(Float32, n_output_channels),
        zeros(Float32, n_output_channels),
        conv.σ,
        conv.pad,
    )
end

"""
Apply convolution, channel modulation and activation.

Args:
    layer: FiLM convolution.
    input: Input tensor `(space, input_channels, batch)`.

Returns:
    Output tensor `(space, output_channels, batch)`.
"""
function (layer::FilmConv)(input)
    # Treat stored base parameters as constants
    weight = Flux.ignore_derivatives() do
        layer.frozen_weight
    end
    bias = Flux.ignore_derivatives() do
        layer.frozen_bias
    end
    output = Flux.conv(input, weight; pad=layer.pad)
    # Shape (1, output_channels, 1) broadcasts across space and batch.
    output = output .+ reshape(bias, 1, :, 1)
    output = reshape(layer.scale, 1, :, 1) .* output .+
        reshape(layer.shift, 1, :, 1)
    return layer.activation.(output)
end

"""
Replace each convolution with a FiLM-adapted layer.

Args:
    base_model: Pretrained model to adapt.

Returns:
    Independent model with the FiLM adaptation.
"""
function build_FiLM(base_model)
    adapted_layers = (FilmConv(deepcopy(layer)) for layer in network(base_model).layers)
    return with_network(base_model, Chain(adapted_layers...))
end

"""
Store a frozen kernel with trainable bias and low-rank factors.

Args:
    frozen_weight: Fixed kernel `(kernel_width, input_channels, output_channels)`.
    bias: Trainable biases.
    left_matrix: Trainable factor `(rank, output_channels)`.
    right_matrix: Trainable factor `(kernel_width, input_channels, rank)`.
    activation: Convolution activation function.
    pad: Convolution padding.

Returns:
    BiasLoRAConv with the base kernel and adaptation parameters.
"""
struct BiasLoRAConv
    frozen_weight::AbstractArray
    bias::AbstractArray
    left_matrix::AbstractArray
    right_matrix::AbstractArray
    activation
    pad
end

Flux.@layer BiasLoRAConv trainable=(bias, left_matrix, right_matrix)

"""
Initialize low-rank adaptation with bias adaptation.

Args:
    convolution: Pretrained convolution to copy.
    rank: Number of low-rank components.
    rng: Random generator for factor initialization.

Returns:
    BiasLoRAConv
"""
function BiasLoRAConv(convolution::Conv, rank::Int; rng=Random.default_rng())
    rank >= 0 || throw(ArgumentError("LoRA rank must be nonnegative"))
    kernel_width, input_channels, output_channels = size(convolution.weight)
    left_matrix = zeros(Float32, rank, output_channels)
    # If rank is zero we only adapt the bias
    if rank > 0
        left_matrix .= randn(rng, Float32, rank, output_channels) .* Float32(sqrt(2 / rank))
    end
    # Zero right factors keep the base kernel unchanged.
    # Random left factors allow initial right-factor gradients.
    right_matrix = zeros(Float32, kernel_width, input_channels, rank)
    return BiasLoRAConv(copy(convolution.weight), copy(convolution.bias),
        left_matrix, right_matrix, convolution.σ, convolution.pad)
end

"""
Store an effective LoRA kernel for one rollout and keep factor gradients.

Args:
    weight: Effective kernel `(kernel_width, input_channels, output_channels)`.
    bias: Output-channel bias vector.
    activation: Convolution activation function.
    pad: Convolution padding.

Returns:
    EffectiveConv keeping the given arrays and activation.
"""
struct EffectiveConv{Weight,Bias,Activation,Padding}
    weight::Weight
    bias::Bias
    activation::Activation
    pad::Padding
end
Flux.@layer EffectiveConv

"""
Apply the materialized kernel, bias and activation.

Args:
    layer: Effective convolution.
    input: Input tensor `(space, input_channels, batch)`.

Returns:
    Output tensor `(space, output_channels, batch)`.
"""
function (layer::EffectiveConv)(input)
    output = Flux.conv(input, layer.weight; pad=layer.pad) .+ reshape(layer.bias, 1, :, 1)
    return layer.activation.(output)
end

"""
Return a non-LoRA layer unchanged.

Args:
    layer: Other type of layer

Returns:
    The original layer.
"""
effective_layer(layer) = layer

"""
Materialize a LoRA kernel while keeping factor gradients.

Args:
    layer: Bias LoRA convolution.

Returns:
    EffectiveConv with the base kernel plus its low-rank update.
"""
function effective_layer(layer::BiasLoRAConv)
    kernel_width, input_channels, output_channels = size(layer.frozen_weight)
    rank = size(layer.left_matrix, 1)
    frozen_weight = Flux.ignore_derivatives() do
        layer.frozen_weight
    end
    # Multiply factors with shapes (kernel_width * input_channels, rank)
    # and (rank, output_channels). Restore the Flux kernel axis order afterward.
    update = reshape(reshape(layer.right_matrix, kernel_width * input_channels, rank) *
        layer.left_matrix, kernel_width, input_channels, output_channels)
    return EffectiveConv(frozen_weight .+ update, layer.bias, layer.activation, layer.pad)
end

"""
Materialize and apply the LoRA kernel. Use effective_model to reuse it across rollout steps.

Args:
    layer: LoRA convolution.
    input: Input tensor `(space, input_channels, batch)`.

Returns:
    Output tensor `(space, output_channels, batch)`.
"""
function (layer::BiasLoRAConv)(input)
    return effective_layer(layer)(input)
end

"""
Replace CNN convolutions with bias-inclusive LoRA layers.

Args:
    base_model: Pretrained model to adapt.
    ranks: One rank per convolution in network order.
    rng: Random generator for factor initialization.

Returns:
    Adapted model.
"""
function build_LoRA(base_model, ranks::AbstractVector; rng=Random.default_rng())
    @assert length(ranks) == length(network(base_model).layers) "Supply one LoRA rank per convolution"
    layers = [BiasLoRAConv(layer, Int(rank); rng=rng)
        for (layer, rank) in zip(network(base_model).layers, ranks)]
    return with_network(base_model, Chain(layers...))
end

"""
Return a model without a supported wrapper unchanged.

Args:
    model: Model

Returns:
    The original model.
"""
effective_model(model) = model

"""
Materialize LoRA kernels once for a rollout. Rebuild after parameter updates.

Args:
    model: Residual or conservative model wrapper.

Returns:
    Wrapper with effective LoRA layers, or the original model when none exist.
"""
function effective_model(model::Union{ResidualBlock,ConservativeVelocityModel})
    if any(layer -> layer isa BiasLoRAConv, network(model).layers)
        original_network = network(model)

        effective_layers = map(effective_layer, original_network.layers)

        new_network = Chain(effective_layers...)

        new_model = with_network(model, new_network)

        return new_model
    end
    return model
end

"""
Create an adaptation model without modifying the pretrained model.

Args:
    pretrained_model: Base model wrapper.
    method: Adaptation method: :full, :output, :first_two, :last_two, :film, or :lora.
    seed: Seed for adaptation initialization.
    ranks: Optional layer-wise LoRA ranks.

Returns:
    Independent model with the requested trainable layers or adaptation factors.
"""
function configure_adaptation(pretrained_model, method; seed, ranks=nothing)
    # FiLM and LoRA use separate adaptation layers.
    if method == :film
        return build_FiLM(pretrained_model)
    elseif method == :lora
        return build_LoRA(pretrained_model, ranks; rng=MersenneTwister(seed))
    end
    # For the other methods, choose which CNN layers can train.
    layer_count = length(network(pretrained_model).layers)
    if method == :full
        trainable_layers = collect(1:layer_count)
    elseif method == :output
        trainable_layers = [layer_count]
    elseif method == :first_two
        trainable_layers = [1, 2]
    elseif method == :last_two
        trainable_layers = [layer_count - 1, layer_count]
    else
        throw(ArgumentError("Unsupported adaptation method: $method"))
    end
    layers = []
    for (index, layer) in enumerate(network(pretrained_model).layers)
        if index in trainable_layers
            # Copy trainable arrays to keep the pretrained model unchanged.
            push!(layers, deepcopy(layer))
        else
            push!(layers, FrozenConv(layer))
        end
    end
    return with_network(pretrained_model, Chain(layers...))
end

"""
Count scalar trainable parameters.

Args:
    model: Model whose trainable arrays are counted.

Returns:
    Total number of scalar entries in Flux.trainables(model).
"""
trainable_parameter_count(model) = sum(length, Flux.trainables(model))
