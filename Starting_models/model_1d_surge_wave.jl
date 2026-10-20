# model_1d_wave_periodic.jl
# 1D linearized shallow-water equation with wind stress
#
#The one-dimensional shallow-water equations can be simplified to a 
# 1D wave-equation with wind forcing:
#
#$\partial h/\partial t + \partial Hu / \partial x = 0$
#
#$\partial u/\partial t + g \partial h / \partial x = \tau/(\rho H)$
#
#where $h$ denotes the water-height above the reference, $u$ the 
#velocity, $D$ the depth below the reference, $H=D+h$ the total water 
#depth and $t,x$ time and space.
#
# The equations and approach are similar to the model_1d_wave_periodic.jl 
# model. Here the equations are not fully linear, on a 'closed' domain,
# and with a wind-stress forcing term.

"""
    Wave1DSurge_cpu
    container for the parameters of the 1D wave equation
    We use a 'functor', i.e. a struct that behaves like a function.
    So this function can contain data, eg parameters.
"""
struct Wave1DSurge_cpu
    g::Float64 # gravity
    D::Vector{Float64} # depth is variable in space
    L::Float64 # length
    W::Float64 # width
    dx::Float64 # spatial step
    nx::Int64 # number of spatial points
    rho::Float64 # density
    C::Float64 # Chezy friction factor
    tau::Function # function of time t: Float64 -> Float64
                    # wind stress as a function of time
    q_left::Function # inflow at left boundary
    q_right::Function # outflow at right boundary
end

"""
    depth_profile(grid_u,D_min,D_max,D_edges,D_widths)
    Create a depth profile with two tanh transitions
    D_min : minimum depth
    D_max : maximum depth
    D_edges : locations of the edges of the transitions
    D_widths : widths of the transitions
"""
function depth_profile(grid_u,D_min,D_max,D_edges,D_widths)
    D = [ D_min +
      (D_max-D_min) * 0.5 * (1 + tanh((x - D_edges[1]) / D_widths[1])) +
      (D_min-D_max) * 0.5 * (1 + tanh((x - D_edges[2]) / D_widths[2]))
        for x in grid_u]
    return D
end

"""
    initial state for the 1D wave equation
    We use ComponentArrays to store the state variables
    x.h : height
    x.u : velocity
    The data is stored on a regular but staggered grid
    h at: dx/2, 3dx/2, 5dx/2, ...
    u at: 0.0, dx, 2dx, ..., L
"""
function initial_state_bump(f::Wave1DSurge_cpu, h0=1.0, w=0.05, c=0.5,u0=0.0)
    x_center=f.L*c
    width=f.L*w
    x_h = f.dx/2:f.dx:(f.L-f.dx/2)
    x_u = 0.0:f.dx:f.L
    h = h0.*exp.(-((x_h .- x_center).^2) ./ (2 * width^2))
    u = u0.*exp.(-((x_u .- x_center).^2) ./ (2 * width^2))
    x = ComponentVector(h=h,u=u)
    return x
end

"""
   Compute the spatial derivative of h
   function dh_dx!(∂h∂x,h,dx)
"""
function dh_dx!(∂h∂x,h,dx)
    nx=length(h)
    for i in 2:nx
        ∂h∂x[i] = (h[i]-h[i-1])/(dx)
    end
    ∂h∂x[1] = 0.0
    ∂h∂x[end] = 0.0
end

"""
   Compute the spatial average of h
   function avg_h!(h_avg,h,dx)
"""
function avg_h!(h_avg,h,dx)
    nx=length(h)
    for i in 2:nx
        h_avg[i] = (h[i]+h[i-1])/2.0
    end
    h_avg[1] = h[1]
    h_avg[end] = h[end]
end

"""
   Compute the spatial derivative of u
   function du_dx!(∂u∂x,u,dx)
"""
function du_dx!(∂u∂x,u,dx)
    nx=length(u)-1
    for i in 1:nx
        ∂u∂x[i] = (u[i+1]-u[i])/(dx)
    end
end

"""
    Wave1DSurge_cpu(dx_dt,x,p,t)
    Compute time derivative of the state
"""
function (f::Wave1DSurge_cpu)(dx_dt,x,p,t)
    g=f.g       # gravity
    D=f.D       # depth profile
    dx=f.dx     # spatial step 
    rho=f.rho   # density
    C=f.C       # Chezy friction factor
    tau_val = f.tau(t) # wind stress
    W=f.W       # width (only used for boundary)
    q_left_val=f.q_left(t) # inflow at left boundary [m^3/s]
    q_right_val=f.q_right(t) # outflow at right boundary
    # temporary variables
    ∂h∂x = similar(x.u) # allocating version, is not optimal for performance
    ∂Hu∂x = similar(x.h)
    h_avg = similar(x.u)
    # compute spatial derivatives and averages
    avg_h!(h_avg,x.h,dx)
    H=h_avg .+ D
    u=x.u
    u[1]=q_left_val/(H[1]*W)
    u[end]=q_right_val/(H[end]*W)
    Hu=H .* u
    du_dx!(∂Hu∂x,Hu,dx)
    dh_dx!(∂h∂x,x.h,dx)
    # compute time derivatives
    @. dx_dt.u = -g * ∂h∂x + tau_val/(rho * H) - (g*x.u*abs(x.u))/(C*C*H)
    dx_dt.u[1]=0.0
    dx_dt.u[end]=0.0
    @. dx_dt.h = -∂Hu∂x
end


struct Wave1DSurgeHiddenDrag{TF, TL, TR, TD}
    g::Float64
    D::Vector{Float64}
    L::Float64
    W::Float64
    dx::Float64
    nx::Int
    rho::Float64
    C_profile::Vector{Float64}
    tau::TF
    q_left::TL
    q_right::TR
    hidden_drag::TD
end

function (f::Wave1DSurgeHiddenDrag)(dx_dt, x, p, t)
    h_avg = similar(x.u)
    flux_gradient = similar(x.h)
    level_gradient = similar(x.u)
    drag = similar(x.u)

    avg_h!(h_avg, x.h, f.dx)
    H = h_avg .+ f.D
    u = x.u
    u[1] = f.q_left(t) / (H[1] * f.W)
    u[end] = f.q_right(t) / (H[end] * f.W)

    du_dx!(flux_gradient, H .* u, f.dx)
    dh_dx!(level_gradient, x.h, f.dx)

    for i in eachindex(u)
        drag[i] = f.hidden_drag(i, t, u[i], H[i])
    end

    tau = f.tau(t)
    @. dx_dt.u = -f.g * level_gradient + tau / (f.rho * H) -
        (f.g * u * abs(u)) / (f.C_profile^2 * H) - drag
    dx_dt.u[1] = 0.0
    dx_dt.u[end] = 0.0
    @. dx_dt.h = -flux_gradient
end
