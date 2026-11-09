using Test, JLD2
include(joinpath(@__DIR__, "..", "src", "Surge.jl"))

@testset "Relative ensemble times" begin
    mktempdir() do directory
        for times in ([100, 110, 120], [0, 10, 20])
            path = joinpath(directory, "ensemble.jld2")
            JLD2.jldsave(path; solution=[:initial, :next, :last], grid=(u=[1, 2],),
                times, params=(D=[30f0, 30f0],))
            dataset = Surge.load_burgers_ensemble("ensemble.jld2"; data_path=directory)
            @test dataset.times == [10.0, 20.0]
            @test eltype(dataset.times) == Float64
            @test dataset.solution == [:initial, :next, :last]
            @test dataset.grid == (u=[1, 2],)
            @test dataset.params == (D=[30f0, 30f0],)
        end
    end
end
