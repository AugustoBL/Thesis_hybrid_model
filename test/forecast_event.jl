using Test
include(joinpath(@__DIR__, "..", "src", "Surge.jl"))

@testset "Autoregressive event forecast" begin
    event = [(h=fill(1f0, 2), u=zeros(Float32, 3), tau=2f0),
             (h=fill(100f0, 2), u=fill(50f0, 3), tau=3f0),
             (h=fill(200f0, 2), u=fill(60f0, 3), tau=4f0)]
    params = (D=fill(30f0, 3),)
    # Increase both state channels by the prescribed stress at each transition.
    model = input -> input[:, 1:2, :] .+ input[:, 4:4, :]
    forecast = Surge.forecast_event(model, event, params)
    @test propertynames(forecast) == (:prediction, :target)
    @test size(forecast.prediction) == (3, 2, 2)
    @test eltype(forecast.prediction) == Float32
    @test forecast.prediction[1:2, 1, 1] == [3f0, 3f0]
    @test forecast.prediction[1:2, 1, 2] == [6f0, 6f0]
    @test forecast.prediction[:, 2, 2] == fill(5f0, 3)
    @test forecast.target[:, :, 1] == Surge._model_target(event[2])
    @test forecast.target[:, :, 2] == Surge._model_target(event[3])
    initial_only = Surge.forecast_event(model, event[1:1], params)
    @test size(initial_only.prediction) == (3, 2, 0)
    @test size(initial_only.target) == (3, 2, 0)
end
