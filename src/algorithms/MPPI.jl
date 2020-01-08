# Algorithm 2 from https://www.cc.gatech.edu/~bboots3/files/InformationTheoreticMPC.pdf
struct MPPI{DT,nu,C<:AbstractMatrix{DT},V,E,F,O,S}
    # MPPI parameters
    K::Int
    H::Int
    covar0::C
    lambda::DT
    gamma::DT
    valuefn::V
    envs::Vector{E} # one per thread
    initfn!::F

    # internal
    noise::Array{DT,3}
    covar_ul::UpperTriangular{DT,C}
    meantrajectory::Matrix{DT}
    trajectorycosts::Vector{DT}
    obsbuffers::Vector{O}
    statebuffers::Vector{S}

    function MPPI{DT}(
        sharedmemory_envctor,
        K::Integer,
        H::Integer,
        covar0::AbstractMatrix{<:Real},
        lambda::Real,
        gamma::Real,
        valuefn,
        initfn!,
    ) where {DT<:AbstractFloat}
        envs = [e for e in sharedmemory_envctor(Threads.nthreads())]

        ssp = statespace(first(envs))
        asp = actionspace(first(envs))
        osp = obsspace(first(envs))

        nd, elt = ndims(asp), eltype(asp)
        if nd != 1 || !(elt <: AbstractFloat)
            error("actionspace(env) must be a vector space of <: AbstractFloat. Got $nd dimensions and eltype $elt.")
        end
        K > 0 || error("K must be > 0. Got $K.")
        H > 0 || error("H must be > 0. Got $H.")
        0 < lambda <= 1 || error("lambda must be 0 < lambda <=1. Got $lambda")
        0 < gamma <= 1 || error("gamma must be 0 < gamma <=1. Got $gamma")
        o = allocate(osp)
        applicable(
            valuefn,
            o,
        ) || error("valuefn must have signature valuefn(::$(typeof(o)))")
        hasmethod(
            initfn!,
            (Matrix{DT},),
        ) || error("initfn! must have signature initfn!(::Matrix{$DT})")

        covar0 = DT.(covar0)
        covar_ul = cholesky(covar0).UL
        meantrajectory = zeros(DT, asp, H)
        trajectorycosts = zeros(DT, K)
        noise = zeros(DT, asp, H, K)
        obsbuffers = [allocate(osp) for _ = 1:Threads.nthreads()]
        statebuffers = [allocate(ssp) for _ = 1:Threads.nthreads()]

        new{
            DT,
            length(asp),
            typeof(covar0),
            typeof(valuefn),
            eltype(envs),
            typeof(initfn!),
            eltype(obsbuffers),
            eltype(statebuffers),
        }(
            K,
            H,
            covar0,
            lambda,
            gamma,
            valuefn,
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

function MPPI(;
    dtype = Float64,
    sharedmemory_envctor,
    covar0,
    lambda,
    K,
    H,
    gamma = 1,
    valuefn = zerofn,
    initfn! = default_initfn!,
)
    MPPI{dtype}(sharedmemory_envctor, K, H, covar0, lambda, gamma, valuefn, initfn!)
end

LyceumBase.reset!(m::MPPI) = (fill!(m.meantrajectory, 0); m)

@propagate_inbounds function LyceumBase.getaction!(
    action::AbstractVector,
    state,
    ::Any,
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
    terminalvalue = convert(DT, m.valuefn(obsbuf))
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
