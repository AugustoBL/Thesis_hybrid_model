module NumericalDataset

using OrdinaryDiffEq, ComponentArrays, JLD2, Random, Statistics
include(joinpath(@__DIR__, "..", "Starting_models", "model_1d_surge_wave.jl"))


"""
Construct the staggered grid, depth profile, and output times.

Args:
    settings: Domain geometry [m], grid size, gravity [m/s^2], depth profile, Chezy coefficient, and sampling settings.

Returns:
    Geometry record with N cell centres, N + 1 velocity faces, face depths [m],
    face Chezy values, grid spacing [m], output interval [s], and sampling times [s].
"""
function dataset_geometry(settings)
    # Water levels use N cell centres; velocities use N + 1 faces including boundaries.
    grid_spacing = settings.domain_length / settings.grid_cells
    velocity_grid = collect(0.0:grid_spacing:settings.domain_length)
    water_level_grid = collect(grid_spacing/2:grid_spacing:settings.domain_length-grid_spacing/2)

    # Convert fractional profile edges and widths to distances [m].
    depth = depth_profile(velocity_grid, settings.minimum_depth, settings.maximum_depth,
        settings.depth_edges .* settings.domain_length, settings.depth_widths .* settings.domain_length)

    # Set output spacing
    output_interval = settings.output_cfl * grid_spacing / sqrt(maximum(depth) * settings.gravity)
    # Include the initial time; the end time is included only if it lies on this sampling grid.
    times = collect(settings.start_time:output_interval:settings.end_time)
    return (grid_spacing=grid_spacing, velocity_grid=velocity_grid, water_level_grid=water_level_grid,
        depth=depth, chezy=fill(settings.chezy_coefficient, length(velocity_grid)),
        output_interval=output_interval, times=times)
end

"""
Compute domain-normalized spatial RMS using trapezoidal weights.

Args:
    values: Real-valued samples `(N + 1,)` on a uniform face grid including both endpoints.
    grid_spacing: Distance between samples [m].
    domain_length: Domain length in the same units as grid_spacing.

Returns:
    Scalar RMS with the units of values.
"""
function spatial_rms(values, grid_spacing, domain_length)
    return sqrt(grid_spacing / domain_length *
        (sum(abs2, values) - 0.5 * first(values)^2 - 0.5 * last(values)^2))
end

"""
Construct hidden linear drag on the velocity grid.

Args:
    geometry: Velocity-face coordinates `(N + 1,)` [m] and grid spacing [m].
    settings: Drag profile, domain length [m], and drag amplitude [1/s].

Returns:
    Face drag vector `(N + 1,)` [1/s], normalized to drag_amplitude for non-control profiles.
"""
function drag_coefficients(geometry, settings)
    coordinate = geometry.velocity_grid ./ settings.domain_length
    if settings.drag_profile == :control
        # The control dataset has no hidden drag.
        return zeros(length(coordinate))
    elseif settings.drag_profile == :uniform
        profile = ones(length(coordinate))
    elseif settings.drag_profile == :interior
        # Peak drag at the domain centre and zero drag at the boundaries.
        profile = sinpi.(coordinate).^2
    else
        @assert settings.drag_profile == :boundary
        # Peak drag at the boundaries and zero drag at the centre.
        profile = cospi.(coordinate).^2
    end

    # Match spatial RMS
    return settings.drag_amplitude .* profile ./
        spatial_rms(profile, geometry.grid_spacing, settings.domain_length)
end

"""
Construct a Gaussian water-level bump with zero initial velocity.

Args:
    geometry: Water-level and velocity grids.
    settings: Bump height [m], centre/width fractions, and domain length [m].

Returns:
    ComponentVector with h `(N,)` [m] and u `(N + 1,)` [m/s]; no water-level padding.
"""
function initial_state(geometry, settings)
    # Convert the bump centre and width from domain fractions to metres.
    center = settings.initial_bump_center * settings.domain_length
    width = settings.initial_bump_width * settings.domain_length
    water_level = settings.initial_bump_height .* exp.(-((geometry.water_level_grid .- center).^2) ./ (2 * width^2))
    # Start all face velocities at rest.
    return ComponentVector(h=water_level, u=zeros(length(geometry.velocity_grid)))
end

"""
Draw a uniform random value within the supplied bounds.

Args:
    random_generator: Random generator for the draw.
    bounds: Lower and upper limits in consistent units.

Returns:
    Sample between the first and last bounds.
"""
function sample_uniform(random_generator, bounds)
    # Rescale a unit-interval draw to the supplied lower and upper bounds.
    return first(bounds) + (last(bounds) - first(bounds)) * rand(random_generator)
end

"""
Sample a wind-stress event.

Args:
    random_generator: Random generator for event parameters.
    settings: Stress amplitudes [Pa], phases [rad], periods/widths [hours or minutes],
        storm timing [hours], and dimensionless storm-shape fractions.

Returns:
    Function mapping time [s] to wind stress [Pa]; parameters are sampled once per event.
"""
function event_wind(random_generator, settings)
    # Sample the background oscillation and convert its period from hours to seconds.
    background_amplitude = sample_uniform(random_generator, settings.background_amplitude)
    background_period = sample_uniform(random_generator, settings.background_period_hours) * 3600
    background_phase = sample_uniform(random_generator, settings.phase)
    # The main storm has separate rise and decay widths [s].
    storm_amplitude = sample_uniform(random_generator, settings.storm_amplitude)
    storm_center = sample_uniform(random_generator, settings.storm_center_hours) * 3600
    rise_width = sample_uniform(random_generator, settings.rise_width_hours) * 3600
    decay_width = sample_uniform(random_generator, settings.decay_width_hours) * 3600
    # Gust oscillations use a period sampled in minutes and converted to seconds.
    gust_period = sample_uniform(random_generator, settings.gust_period_minutes) * 60
    gust_phase = sample_uniform(random_generator, settings.phase)
    # Scale the secondary storm by the primary amplitude and delay it relative to the primary centre.
    second_amplitude = storm_amplitude * sample_uniform(random_generator, settings.second_amplitude_fraction)
    second_center = storm_center + sample_uniform(random_generator, settings.second_delay_hours) * 3600
    second_width = sample_uniform(random_generator, settings.second_width_hours) * 3600

    return function(time)
        width = time <= storm_center ? rise_width : decay_width
        primary_envelope = exp(-0.5 * ((time - storm_center) / width)^2)
        secondary_envelope = exp(-0.5 * ((time - second_center) / second_width)^2)
        background = background_amplitude * sin(2pi * time / background_period + background_phase)
        primary = storm_amplitude * primary_envelope *
            (settings.storm_mean_fraction + settings.storm_gust_fraction * sin(2pi * time / gust_period + gust_phase))
        return background + primary + second_amplitude * secondary_envelope
    end
end

"""
Sample a numerical solution and its wind forcing into event records.

Args:
    solution: Numerical solution callable at output times.
    wind: Wind-stress function of time [s].
    times: Output sampling times [s].

Returns:
    One Float32 ComponentVector per sampling time, including the initial time when supplied:
    h `(N,)` [m], u `(N + 1,)` [m/s], scalar tau [Pa], and zero q_left/q_right [m^3/s].
"""
function pack_event(solution, wind, times)
    # Sample states and forcing at identical times in the format expected by the data loader.
    return [ComponentVector(h=Float32.(solution(time).h), u=Float32.(solution(time).u),
        tau=Float32(wind(time)), q_left=0f0, q_right=0f0) for time in times]
end

"""
Generate and save a surge ensemble for a specified drag profile.

Args:
    output_file: Destination JLD2 filepath.
    settings: Geometry, physical parameters, initial state, drag profile, event count, time span, and seed.
    wind_settings: Wind-event sampling settings.
    solver: OrdinaryDiffEq integration algorithm.
    absolute_tolerance: Absolute solver tolerance for the h/u state.
    relative_tolerance: Relative solver tolerance.

Returns:
    Output filepath after saving event trajectories, sampling times [s], cell/face grids,
    physical parameters, hidden face drag, and generation settings.
"""
function generate_dataset(output_file; settings, wind_settings, solver, absolute_tolerance, relative_tolerance)
    geometry = dataset_geometry(settings)
    drag = drag_coefficients(geometry, settings)
    # One seeded generator gives reproducible wind draws
    random_generator = MersenneTwister(settings.seed)
    events = []
    for event_index in 1:settings.event_count
        # Draw a new wind signal
        wind = event_wind(random_generator, wind_settings)

        hidden_drag(position, time, velocity, depth) = drag[position] * velocity
        # Both boundaries are closed: discharge [m^3/s] is zero at every time.
        zero_discharge(time) = 0.0
        # Pass face depths, Chezy values, wind, boundary discharges, and hidden drag to the solver model.
        model = Wave1DSurgeHiddenDrag(settings.gravity, geometry.depth, settings.domain_length,
            settings.domain_width, geometry.grid_spacing, settings.grid_cells, settings.water_density,
            geometry.chezy, wind, zero_discharge, zero_discharge, hidden_drag)
        # Each event starts from the same Gaussian bump and zero velocity.
        problem = ODEProblem(model, initial_state(geometry, settings), (settings.start_time, settings.end_time))

        # Save at the common output times while the solver controls its internal timesteps.
        solution = solve(problem, solver; saveat=geometry.times,
            abstol=absolute_tolerance, reltol=relative_tolerance)
        # Store Float32 states and wind stress, retaining the initial state.
        push!(events, pack_event(solution, wind, geometry.times))
    end

    # Keep base-model parameters saved below.
    parameters = (g=settings.gravity, C=settings.chezy_coefficient, C_profile=geometry.chezy,
        rho=settings.water_density, W=settings.domain_width, D=geometry.depth,
        L=settings.domain_length, dx=geometry.grid_spacing, nx=settings.grid_cells, linear_drag=0.0)
    # Save the data and settings needed to identify and reproduce this ensemble.
    mkpath(dirname(output_file))
    JLD2.jldsave(output_file; solution=events, times=geometry.times, params=parameters,
        grid=ComponentVector(h=geometry.water_level_grid, u=geometry.velocity_grid),
        settings=settings, wind_settings=wind_settings, drag_coefficients=drag,
        absolute_tolerance=absolute_tolerance, relative_tolerance=relative_tolerance,
        solver=string(solver))
    return output_file
end

end
