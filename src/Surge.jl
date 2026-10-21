module Surge

using BSON, ComponentArrays, Flux, JLD2, Random, Statistics


include("models.jl")
include("data_utils.jl")
include("training_utils.jl")
include("evaluation_utils.jl")
include("experiment_utils.jl")

export ResidualBlock, FilmConv, BiasLoRAConv, build_base_cnn, build_FiLM, build_LoRA
export ConservativeVelocityModel, build_conservative_velocity_cnn, build_flux_correction_cnn
export network, with_network
export configure_adaptation, trainable_parameter_count, load_model, save_model
export load_burgers_ensemble, get_data_path, get_model_path
export SparseStationEventDataset, prepare_sparse_station_event_data, add_observation_noise
export make_event_splits, train_model!, run_experiment, forecast_event, evaluate_run

end
