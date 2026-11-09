module ExperimentPlots

using Plots

"""
Plot adapted and pretrained MSE against position and forecast time.

Args:
    run: Run record with an existing evaluation directory.
    evaluation: MSE maps `(N + 1, 2, steps)`, N cell/N + 1 face coordinates [m] and times [s].

Returns:
    Figure with pretrained/adapted rows and h/u columns, saved as evaluation/mse_heatmaps.png.
"""
function plot_mse_heatmaps(run, evaluation)
    panels = []
    # Add pretrained h/u panels first, then adapted h/u panels.
    for (label, errors) in (("Pretrained", evaluation.baseline_mse), ("Adapted", evaluation.mse))
        # Use N cells for h and all N + 1 faces for u.
        for (channel, grid, variable) in ((1, evaluation.grid.h, "h"), (2, evaluation.grid.u, "u"))
            # Heatmap rows are positions. Columns are forecast times in hours.
            push!(panels, heatmap(evaluation.times ./ 3600, grid, errors[1:length(grid), channel, :];
                xlabel="Forecast [hours]", ylabel="Position [m]", title="$label $variable MSE"))
        end
    end
    # Keep h errors [m^2] and u errors [m^2/s^2] in separate panels.
    figure = plot(panels...; layout=(2, 2), size=(1100, 700))
    savefig(figure, joinpath(run.directory, "evaluation", "mse_heatmaps.png"))
    return figure
end

"""
Animate reference, pretrained and adapted forecast profiles.

Args:
    run: Run record with an existing evaluation directory.
    evaluation: Saved prediction, baseline and target `(N + 1, 2, steps)`, grids [m] and times [s].
    frames_per_second: Animation playback rate.
    maximum_frames: Maximum number of sampled forecast frames.

Returns:
    Plots.gif result saved as evaluation/forecast.gif.
"""
function forecast_gif(run, evaluation; frames_per_second, maximum_frames)
    frame_indices = unique(round.(Int, range(1, length(evaluation.times);
        length=min(maximum_frames, length(evaluation.times)))))
    animation = Animation()
    # Reuse the first saved test forecast without running the models again.
    forecast = evaluation.forecast
    for step in frame_indices
        panels = []
        for (channel, grid, variable) in ((1, evaluation.grid.h, "h [m]"), (2, evaluation.grid.u, "u [m/s]"))
            # Use N cells for h and N + 1 faces for u. Leave out h padding.
            indices = eachindex(grid)
            # Show the numerical reference and elapsed forecast time for this frame.
            panel = plot(grid, forecast.target[indices, channel, step]; label="Reference", ylabel=variable,
                title="Forecast $(round(evaluation.times[step] / 3600; digits=2)) hours")
            # Compare both surrogate predictions at the same positions and time.
            plot!(panel, grid, forecast.baseline[indices, channel, step]; label="Pretrained", linestyle=:dash)
            plot!(panel, grid, forecast.prediction[indices, channel, step]; label="Adapted")
            push!(panels, panel)
        end
        # Stack h above u and record one frame for this forecast step.
        frame(animation, plot(panels...; layout=(2, 1), size=(850, 600)))
    end
    # Playback rate affects animation duration, not the underlying forecast times.
    return gif(animation, joinpath(run.directory, "evaluation", "forecast.gif"); fps=frames_per_second)
end

"""
Plot finite training and validation losses.

Args:
    run: Run record with epoch/loss history and an existing evaluation directory.

Returns:
    Two-panel figure saved as evaluation/training_validation_loss.png.
"""
function plot_losses(run)
    # Omit epoch-zero training NaN and any other nonfinite training entries.
    training = filter(row -> isfinite(row.training_loss), run.result.history)
    # Skip epochs without a finite validation score.
    validation = filter(row -> isfinite(row.validation_loss), run.result.history)
    # Use separate panels because training and validation losses can have different scales.
    training_plot = plot([row.epoch for row in training], [row.training_loss for row in training];
        xlabel="Epoch", ylabel="Training objective", label="Training")
    validation_plot = plot([row.epoch for row in validation], [row.validation_loss for row in validation];
        xlabel="Epoch", ylabel="Station MSE", label="Validation")
    # Use recorded epoch numbers so validation points retain their actual spacing.
    figure = plot(training_plot, validation_plot; layout=(2, 1), size=(850, 650))
    savefig(figure, joinpath(run.directory, "evaluation", "training_validation_loss.png"))
    return figure
end

"""
Plot diagnostic time series for each saved model and event.

Args:
    run_directory: Run directory with an existing diagnostics subdirectory.
    diagnostics: Model/event records with one metric record per elapsed time [s].

Returns:
    Six-panel figure saved as diagnostics/report_series.png.
"""
function plot_report_series(run_directory, diagnostics)
    panels = []
    # Show one diagnostic per panel. Volume drift is [m^3] and amplitude error is [m].
    for (metric, label) in ((:mse_h, "h MSE"), (:mse_u, "u MSE"),
        (:bias_squared, "Squared mean bias"), (:anomaly_mse, "Anomaly MSE"),
        (:amplitude_error, "Amplitude error"), (:volume_drift, "Volume drift"))
        panel = plot(; xlabel="Forecast [hours]", ylabel=label)
        # Use one curve per model and event to show differences between events.
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
