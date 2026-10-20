module ExperimentPlots

using Plots

"""
Plot adapted and pretrained spatial-time MSE maps.

Args:
    run: Run record with an evaluation directory.
    evaluation: Evaluation result with MSE maps, grids [m], and times [s].

Returns:
    Four-panel figure saved as evaluation/mse_heatmaps.png.
"""
function plot_mse_heatmaps(run, evaluation)
    panels = []
    for (label, errors) in (("Pretrained", evaluation.baseline_mse), ("Adapted", evaluation.mse))
        # Use each variable's physical grid length so water-level padding cannot
        # appear as a spurious low-error strip.
        for (channel, grid, variable) in ((1, evaluation.grid.h, "h"), (2, evaluation.grid.u, "u"))
            push!(panels, heatmap(evaluation.times ./ 3600, grid, errors[1:length(grid), channel, :];
                xlabel="Forecast [hours]", ylabel="Position [m]", title="$label $variable MSE"))
        end
    end
    # Keep model and physical-variable comparisons in separate panels; h
    # and u squared errors have different units.
    figure = plot(panels...; layout=(2, 2), size=(1100, 700))
    savefig(figure, joinpath(run.directory, "evaluation", "mse_heatmaps.png"))
    return figure
end

"""
Animate reference, pretrained, and adapted forecast profiles.

Args:
    run: Run record with an evaluation directory.
    evaluation: Evaluation result containing one forecast `(space, 2, steps)`, grids, and times.
    frames_per_second: Animation playback rate.
    maximum_frames: Maximum number of sampled forecast frames.

Returns:
    Plots.gif result saved as evaluation/forecast.gif.
"""
function forecast_gif(run, evaluation; frames_per_second, maximum_frames)
    # Spread frames across the full available forecast, round to valid sample
    # indices, and remove duplicates introduced by rounding.
    frame_indices = unique(round.(Int, range(1, length(evaluation.times);
        length=min(maximum_frames, length(evaluation.times)))))
    animation = Animation()
    # Reuse the first saved test-event forecast rather than recomputing any model.
    forecast = evaluation.forecast
    for step in frame_indices
        panels = []
        for (channel, grid, variable) in ((1, evaluation.grid.h, "h [m]"), (2, evaluation.grid.u, "u [m/s]"))
            # Trim each channel to its own staggered-grid length before plotting.
            indices = eachindex(grid)
            panel = plot(grid, forecast.target[indices, channel, step]; label="Reference", ylabel=variable,
                title="Forecast $(round(evaluation.times[step] / 3600; digits=2)) hours")
            # Overlay the pretrained reference on the same coordinates and time as
            # the adapted prediction and numerical target.
            plot!(panel, grid, forecast.baseline[indices, channel, step]; label="Pretrained", linestyle=:dash)
            plot!(panel, grid, forecast.prediction[indices, channel, step]; label="Adapted")
            push!(panels, panel)
        end
        frame(animation, plot(panels...; layout=(2, 1), size=(850, 600)))
    end
    return gif(animation, joinpath(run.directory, "evaluation", "forecast.gif"); fps=frames_per_second)
end

"""
Plot finite training and validation losses.

Args:
    run: Run record containing history and an evaluation directory.

Returns:
    Two-panel figure saved as evaluation/training_validation_loss.png.
"""
function plot_losses(run)
    # Epoch-zero training loss is normally NaN; omit it and any nonfinite
    # entries rather than plotting misleading zeros.
    training = filter(row -> isfinite(row.training_loss), run.result.history)
    # Retain only actual finite validation measurements, which may be less
    # frequent than training-history entries.
    validation = filter(row -> isfinite(row.validation_loss), run.result.history)
    training_plot = plot([row.epoch for row in training], [row.training_loss for row in training];
        xlabel="Epoch", ylabel="Training objective", label="Training")
    validation_plot = plot([row.epoch for row in validation], [row.validation_loss for row in validation];
        xlabel="Epoch", ylabel="Station MSE", label="Validation")
    figure = plot(training_plot, validation_plot; layout=(2, 1), size=(850, 650))
    savefig(figure, joinpath(run.directory, "evaluation", "training_validation_loss.png"))
    return figure
end

"""
Plot diagnostic time series for each saved model and event.

Args:
    run_directory: Run directory containing diagnostics outputs.
    diagnostics: Diagnostic series and times [s] returned by diagnose_saved_run.

Returns:
    Six-panel figure saved as diagnostics/report_series.png.
"""
function plot_report_series(run_directory, diagnostics)
    panels = []
    for (metric, label) in ((:mse_h, "h MSE"), (:mse_u, "u MSE"),
        (:bias_squared, "Squared mean bias"), (:anomaly_mse, "Anomaly MSE"),
        (:amplitude_error, "Amplitude error"), (:volume_drift, "Volume drift"))
        panel = plot(; xlabel="Forecast [hours]", ylabel=label)
        # Preserve event-to-event variation rather than collapsing all curves
        # into a single average that could hide a failing event.
        for record in diagnostics.series
            plot!(panel, diagnostics.times ./ 3600,
                [getproperty(row, metric) for row in record.series]; label="$(record.model), event $(record.event)")
        end
        push!(panels, panel)
    end
    figure = plot(panels...; layout=(3, 2), size=(1200, 1000))
    savefig(figure, joinpath(run_directory, "diagnostics", "report_series.png"))
    return figure
end

end
