module NumericalDataset

using OrdinaryDiffEq, ComponentArrays, JLD2, Random, Statistics
include(joinpath(@__DIR__, "..", "Starting_models", "model_1d_surge_wave.jl"))


"""
Construct the staggered grid, depth profile, and output times.

Args:
    settings: Domain geometry [m], grid size, gravity [m/s^2], depth profile, Chezy coefficient, and sampling settings.

Returns:
    Geometry record with cell/face coordinates, depths, Chezy values, and output times [s].
"""
function dataset_geometry(settings)
    # N cells have N+1 velocity faces and N water-level centres.
    grid_spacing = settings.domain_length / settings.grid_cells
    velocity_grid = collect(0.0:grid_spacing:settings.domain_length)
    water_level_grid = collect(grid_spacing/2:grid_spacing:settings.domain_length-grid_spacing/2)

    depth = depth_profile(velocity_grid, settings.minimum_depth, settings.maximum_depth,
        settings.depth_edges .* settings.domain_length, settings.depth_widths .* settings.domain_length)

    output_interval = settings.output_cfl * grid_spacing / sqrt(maximum(depth) * settings.gravity)
    times = collect(settings.start_time:output_interval:settings.end_time)
    return (grid_spacing=grid_spacing, velocity_grid=velocity_grid, water_level_grid=water_level_grid,
        depth=depth, chezy=fill(settings.chezy_coefficient, length(velocity_grid)),
        output_interval=output_interval, times=times)
end

"""
Compute domain-normalized spatial RMS using trapezoidal weights.

Args:
    values: Samples on a uniform grid including both endpoints.
    grid_spacing: Distance between samples.
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
    geometry: Grid coordinates and spacing.
    settings: Drag profile, domain length [m], and drag amplitude [1/s].

Returns:
    Face drag coefficients [1/s] with the prescribed spatial RMS, or zeros for control.
"""
function drag_coefficients(geometry, settings)
    coordinate = geometry.velocity_grid ./ settings.domain_length
    if settings.drag_profile == :control
        return zeros(length(coordinate))
    elseif settings.drag_profile == :uniform
        profile = ones(length(coordinate))
    elseif settings.drag_profile == :interior
        profile = sinpi.(coordinate).^2
    else
        @assert settings.drag_profile == :boundary
        profile = cospi.(coordinate).^2
    end

    return settings.drag_amplitude .* profile ./
        spatial_rms(profile, geometry.grid_spacing, settings.domain_length)
end

"""
Construct a Gaussian water-level bump with zero initial velocity.

Args:
    geometry: Water-level and velocity grids.
    settings: Bump height [m], centre/width fractions, and domain length [m].

Returns:
    ComponentVector with N water levels [m] and N + 1 face velocities [m/s].
"""
function initial_state(geometry, settings)
    center = settings.initial_bump_center * settings.domain_length
    width = settings.initial_bump_width * settings.domain_length
    water_level = settings.initial_bump_height .* exp.(-((geometry.water_level_grid .- center).^2) ./ (2 * width^2))
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
    return first(bounds) + (last(bounds) - first(bounds)) * rand(random_generator)
end

"""
Sample a wind-stress event.

Args:
    random_generator: Random generator for event parameters.
    settings: Stress amplitudes [Pa], phases [rad], durations, and storm-shape fractions.

Returns:
    Function mapping time in seconds to wind stress [Pa].
"""
function event_wind(random_generator, settings)
    background_amplitude = sample_uniform(random_generator, settings.background_amplitude)
    background_period = sample_uniform(random_generator, settings.background_period_hours) * 3600
    background_phase = sample_uniform(random_generator, settings.phase)
    storm_amplitude = sample_uniform(random_generator, settings.storm_amplitude)
    storm_center = sample_uniform(random_generator, settings.storm_center_hours) * 3600
    rise_width = sample_uniform(random_generator, settings.rise_width_hours) * 3600
    decay_width = sample_uniform(random_generator, settings.decay_width_hours) * 3600
    gust_period = sample_uniform(random_generator, settings.gust_period_minutes) * 60
    gust_phase = sample_uniform(random_generator, settings.phase)
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
    Float32 state records containing h [m], u [m/s], stress [Pa], and zero boundary discharges.
"""
function pack_event(solution, wind, times)
    return [ComponentVector(h=Float32.(solution(time).h), u=Float32.(solution(time).u),
        tau=Float32(wind(time)), q_left=0f0, q_right=0f0) for time in times]
end

"""
Generate and save a surge ensemble for a specified drag profile.

Args:
    output_file: Destination JLD2 filepath; existing output is overwritten.
    settings: Geometry, physical parameters, initial state, drag profile, event count, time span, and seed.
    wind_settings: Wind-event sampling settings.
    solver: OrdinaryDiffEq integration algorithm.
    absolute_tolerance: Absolute solver tolerance for the h/u state.
    relative_tolerance: Relative solver tolerance.

Returns:
    Output filepath after saving trajectories, grids, parameters, and generation metadata.
"""
function generate_dataset(output_file; settings, wind_settings, solver, absolute_tolerance, relative_tolerance)
    geometry = dataset_geometry(settings)
    drag = drag_coefficients(geometry, settings)
    random_generator = MersenneTwister(settings.seed)
    events = []
    for event_index in 1:settings.event_count
        wind = event_wind(random_generator, wind_settings)

        """
        Compute the hidden linear-drag term at one velocity face.

        Args:
            position: Velocity-face index.
            time: Simulation time [s].
            velocity: Face velocity [m/s].
            depth: Face depth [m].

        Returns:
            Linear-drag term [m/s^2].
        """
        hidden_drag(position, time, velocity, depth) = drag[position] * velocity
        """
        Supply zero boundary discharge.

        Args:
            time: Simulation time [s].

        Returns:
            Zero discharge [m^3/s].
        """
        zero_discharge(time) = 0.0
        model = Wave1DSurgeHiddenDrag(settings.gravity, geometry.depth, settings.domain_length,
            settings.domain_width, geometry.grid_spacing, settings.grid_cells, settings.water_density,
            geometry.chezy, wind, zero_discharge, zero_discharge, hidden_drag)
        problem = ODEProblem(model, initial_state(geometry, settings), (settings.start_time, settings.end_time))

        solution = solve(problem, solver; saveat=geometry.times,
            abstol=absolute_tolerance, reltol=relative_tolerance)
        @assert string(solution.retcode) in ("Success", "ReturnCode.Success") "Numerical simulation failed"
        push!(events, pack_event(solution, wind, geometry.times))
    end

    parameters = (g=settings.gravity, C=settings.chezy_coefficient, C_profile=geometry.chezy,
        rho=settings.water_density, W=settings.domain_width, D=geometry.depth,
        L=settings.domain_length, dx=geometry.grid_spacing, nx=settings.grid_cells, linear_drag=0.0)
    mkpath(dirname(output_file))
    JLD2.jldsave(output_file; solution=events, times=geometry.times, params=parameters,
        grid=ComponentVector(h=geometry.water_level_grid, u=geometry.velocity_grid),
        settings=settings, wind_settings=wind_settings, drag_coefficients=drag,
        absolute_tolerance=absolute_tolerance, relative_tolerance=relative_tolerance,
        solver=string(solver))
    return output_file
end

end
