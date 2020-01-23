struct MPPI{DT<:AbstractFloat,nu,Covar<:AbsMat{DT},Value,Env,Init,Obs,State}
    # MPPI parameters
    K::Int
    H::Int
    lambda::DT
    gamma::DT
    value::Value
    envs::Vector{Env} # one per thread
    initfn!::Init

    # internal
    noise::Array{DT,3}
    covar_ul::Covar
    meantrajectory::Matrix{DT}
    trajectorycosts::Vector{DT}
    obsbuffers::Vector{Obs}
    statebuffers::Vector{State}

    function MPPI{DT}(
        env_tconstructor,
        K::Integer,
        H::Integer,
        covar::AbstractMatrix{<:Real},
        lambda::Real,
        gamma::Real,
        value,
        initfn!,
    ) where {DT<:AbstractFloat}
        envs = [e for e in env_tconstructor(Threads.nthreads())]

        ssp = statespace(first(envs))
        asp = actionspace(first(envs))
        osp = obsspace(first(envs))

        if !(asp isa Shapes.AbstractVectorShape)
            throw(ArgumentError("actionspace(env) must be a Shape.AbstractVectorShape"))
        end
        K > 0 || error("K must be > 0. Got $K.")
        H > 0 || error("H must be > 0. Got $H.")
        0 < lambda <= 1 || throw(ArgumentError("lambda must be in interval (0, 1]"))
        0 < gamma <= 1 || throw(ArgumentError("gamma must be in interval (0, 1]"))

        covar_ul = convert(AbsMat{DT}, cholesky(covar).UL)
        meantrajectory = zeros(DT, asp, H)
        trajectorycosts = zeros(DT, K)
        noise = zeros(DT, asp, H, K)
        obsbuffers = [allocate(osp) for _ = 1:Threads.nthreads()]
        statebuffers = [allocate(ssp) for _ = 1:Threads.nthreads()]

        new{
            DT,
            length(asp),
            typeof(covar_ul),
            typeof(value),
            eltype(envs),
            typeof(initfn!),
            eltype(obsbuffers),
            eltype(statebuffers),
        }(
            K,
            H,
            lambda,
            gamma,
            value,
            envs,
            initfn!,
            noise,
            covar_ul,
            meantrajectory,
            trajectorycosts,
            obsbuffers,
            statebuffers
        )
    end
end

"""
    $(TYPEDEF)

    MPPI{DT<:AbstractFloat}(args...; kwargs...) -> MPPI
    MPPI(args...; kwargs...) -> MPPI

Construct an instance of  `MPPI` with `args` and `kwargs`, where `DT <: AbstractFloat` is
the element type used for pre-allocated buffers, which defaults to Float32.

In the following explanation of the `MPPI` constructor, we use the
following notation:
- `U::Matrix`: the canonical control vector ``(u_{1}, u_{2}, \\dots, u_{H})``, where
    `size(U) == (length(actionspace(env)), H)`.

# Keywords

- `env_tconstructor`: a function with signature `env_tconstructor(n)` that returns `n`
    instances of `T`, where `T <: AbstractEnvironment`.
- `H::Integer`: Length of sampled trajectories.
- `K::Integer`: Number of trajectories to sample.
- `covar::AbstractMatrix`: The covariance matrix for the Normal distribution from which
    control pertubations are sampled from.
- `gamma::Real`: Reward discount, applied as `gamma^(t - 1) * reward[t]`.
- `lambda::Real`: Temperature parameter for the exponential reweighting of sampled
    trajectories. In the limit that lambda approaches 0, `U` is set to the highest reward
    trajectory. Conversely, as `lambda` approaches infinity, `U` is computed as the
    unweighted-average of the samples trajectories.
- `value`: a function mapping observations to scalar rewards, with the signature
    `value(obs::AbstractVector) --> reward::Real`
- `initfn!`: A function with the signature `initfn!(U::Matrix)` used for
    re-initializing `U` after shifting it. Defaults to setting the last
    element of `U` to 0.
"""
function MPPI{DT}(;
    env_tconstructor,
    covar,
    lambda,
    K,
    H,
    gamma = 1,
    value = zerofn,
    initfn! = default_initfn!,
) where {DT<:AbstractFloat}
    MPPI{DT}(env_tconstructor, K, H, covar, lambda, gamma, value, initfn!)
end

MPPI(args...; kwargs...) = MPPI{Float32}(args...; kwargs...)

"""
    $(TYPEDSIGNATURES)

Resets the canonical control vector to zeros.
"""
LyceumBase.reset!(m::MPPI) = (fill!(m.meantrajectory, 0); m)

"""
    $(SIGNATURES)

Starting from the environment's `state`, perform one step of the MPPI algorithm and
store the resulting action in `action`. The trajectory sampling portion of MPPI is
done in parallel using `nthreads` threads.
"""
@propagate_inbounds function LyceumBase.getaction!(
    action::AbstractVector,
    state,
    m::MPPI{DT,nu};
    nthreads::Integer = Threads.nthreads(),
) where {DT,nu}
    @boundscheck begin
        length(action) == nu || throw(ArgumentError("Expected action vector of length $nu. Got: $(length(action))"))
        require_one_based_indexing(action)
    end

    nthreads = min(m.K, nthreads)
    step!(m, state, nthreads)
    @inbounds copyto!(action, uview(m.meantrajectory, :, 1))
    shiftcontrols!(m)
    return action
end

function LyceumBase.step!(m::MPPI{DT,nu}, s, nthreads) where {DT,nu}
    randn!(m.noise)
    lmul!(m.covar_ul, reshape(m.noise, (nu, :)))
    if nthreads == 1
        # short circuit
        threadstep!(m, s, 1:m.K)
    else
        kranges = splitrange(m.K, nthreads)
        @sync for i = 1:nthreads
            Threads.@spawn threadstep!(m, s, kranges[i])
        end
    end
    combinetrajectories!(m)
end

function threadstep!(m::MPPI, s, krange)
    tid = Threads.threadid()
    for k in krange
        perturbedrollout!(m, s, k, tid)
    end
end

function perturbedrollout!(m::MPPI{DT,nu}, state, k, tid) where {DT,nu}
    env = m.envs[tid]
    obsbuf = m.obsbuffers[tid]
    statebuf = m.statebuffers[tid]
    mean = m.meantrajectory
    noise = m.noise

    setstate!(env, state)
    discountedreward = zero(DT)
    discountfactor = one(DT)
    @uviews mean noise @inbounds for t = 1:m.H
        mean_t = SVector{nu,DT}(view(mean, :, t))
        noise_tk = SVector{nu,DT}(view(noise, :, t, k))
        action_t = mean_t + noise_tk
        setaction!(env, action_t)

        step!(env)

        getobs!(obsbuf, env)
        getstate!(statebuf, env)
        reward = getreward(statebuf, action_t, obsbuf, env)

        discountedreward += reward * discountfactor
        discountfactor *= m.gamma
    end # env at t=H+1
    getobs!(obsbuf, env)
    terminalvalue = convert(DT, m.value(obsbuf))
    @inbounds m.trajectorycosts[k] = -(discountedreward + terminalvalue * discountfactor)
    return m
end

function combinetrajectories!(m::MPPI{DT,nu}) where {DT,nu}
    costs = m.trajectorycosts

    beta = minimum(costs)
    eta = zero(DT)
    for k = 1:m.K
        @inbounds costs[k] = softcost = exp((beta - costs[k]) / m.lambda)
        eta += softcost
    end

    costs ./= eta

    for k = 1:m.K, t = 1:m.H, u = 1:nu
        @inbounds m.meantrajectory[u, t] += costs[k] * m.noise[u, t, k]
    end

    m
end

function shiftcontrols!(m::MPPI{DT,nu}) where {DT,nu}
    for t = 2:m.H, u = 1:nu
        @inbounds m.meantrajectory[u, t-1] = m.meantrajectory[u, t]
    end
    m.initfn!(m.meantrajectory)
    m
end

@inline function default_initfn!(meantraj)
    @uviews meantraj @inbounds begin
        lastcontrols = view(meantraj, :, size(meantraj, 2))
        fill!(lastcontrols, 0)
    end
end
