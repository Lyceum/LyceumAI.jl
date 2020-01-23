struct NaturalPolicyGradient{DT<:AbstractFloat,S,P,V,VF,VOP}
    envsampler::S
    policy::P

    value::V
    valuefit!::VF
    value_feature_op::VOP

    Hmax::Int
    N::Int
    Nmean::Int

    norm_step_size::DT
    gamma::DT
    gaelambda::DT
    whiten_advantages::Bool
    bootstrapped_nstep_returns::Bool

    max_cg_iter::Int
    cg_tol::DT

    vanilla_pg::Vector{DT} # nparams
    natural_pg::Vector{DT} # nparams

    fvp_op::FVP{DT, Array{DT, 2}}
    cg_op::CG{DT}

    advantages_vec::Vector{DT} # N
    returns_vec::Vector{DT} # N
end

"""
    NaturalPolicyGradient{DT<:AbstractFloat}(args...; kwargs...) -> NaturalPolicyGradient
    NaturalPolicyGradient(args...; kwargs...) -> NaturalPolicyGradient

Construct a `NaturalPolicyGradient` with `args` and `kwargs` using `DT <: AbstractFloat` as
the element type for pre-allocated buffers, which defaults to Float32.

In the following explanation of the `NaturalPolicyGradient` constructor, we use the
following notation/shorthands:
- `dim_o = length(obsspace(env))`
- `dim_a = length(actionspace(env))`
- "terminal" (e.g. terminal observation) refers to timestep `T + 1` for a length `T` trajectory.

# Arguments

- `env_tconstructor`: a function with signature `env_tconstructor(n)` that returns `n`
    instances of `T`, where `T <: AbstractEnvironment`.
- `policy`: a function mapping observations to actions, with the following signatures:
    - `policy(obs::AbstractVector)` --> `action::AbstractVector`,
        where `size(obs) == (dim_o, )` and `size(action) == (dim_a, )`.
    - `policy(obs::AbstractMatrix)` --> `action::AbstractMatrix`,
        where `size(obs) == (dim_o, N)` and `size(action) == (dim_a, N)`.
- `value`: a function mapping observations to scalar rewards, with the following signatures:
    - `value(obs::AbstractVector)` --> `reward::AbstractVector`,
        where `size(obs) == (dim_o, )` and `size(reward) == (1, )`.
    - `value(obs::AbstractMatrix)` --> `reward::AbstractVector`,
        where `size(obs) == (dim_o, N)` and `size(reward) == (1, N)`.
- `valuefit!`: a function with signature `valuefit!(value, obs::AbstractMatrix, returns::AbstractVector)`,
    where `size(obs) == (dim_o, N)` and `size(returns) == (1, N)`, that fits `value` to
    `obs` and `returns`.

# Keywords

- `Hmax::Integer`: Maximum trajectory length for environments rollouts.
- `N::Integer`: Total number of data samples used for each policy gradient step.
- `Nmean::Integer`: Total number of data samples for the mean policy (without stochastic
    noise). Mean rollouts are used for evaluating `policy` and not used to improve `policy`
    in any form.
- `norm_step_size::Real`: Scaling for the applied gradient update after gradient normalization
    has occured. This process makes training much more stable to step sizes;
    see equation 5 in [this paper](https://arxiv.org/pdf/1703.02660.pdf) for more details.
- `gamma::Real`: Reward discount, applied as `gamma^(t - 1) * reward[t]`.
- `gaelambda::Real`: Generalized Advantage Estimate parameter, balances bias and variance when
    computing advantages. See [this paper](https://arxiv.org/pdf/1506.02438.pdf) for details.
- `max_cg_iter::Integer`: Maximum number of Conjugate Gradient iterations when estimating
    `natural_gradient = alpha * inv(FIM) * gradient`, where `FIM` is the Fisher Information Matrix.
- `cg_tol::Real`: Numerical tolerance for Conjugate Gradient convergence.
- `whiten_advantages::Bool`: if `true`, apply statistical whitening to calculated advantages
    (resulting in `mean(returns) ≈ 0 && std(returns) ≈ 1`).
- `bootstrapped_nstep_returns::Bool`: if `true`, bootstrap the returns calculation starting
    `value(terminal_observation)` instead of 0. See "Reinforcement Learning" by
    Sutton & Barto for further information.
- `value_feature_op`: a function with the below signatures that transforms environment
    observations to a set of "features" to be consumed by `value` and `valuefit!`:
    - `value_feature_op(observations::AbstractVector{<:AbstractMatrix}) --> AbstractMatrix`
    - `value_feature_op(terminal_observations::AbstractMatrix, trajlengths::Vector{<:Integer}) --> AbstractMatrix`,
        where `observations` is a vector of observations from each trajectory,
        `terminal_observations` has size `(dim_o, number_of_trajectories)`, and `trajlengths`
        contains the lengths of each trajectory
        (such that `trajlengths[i] == size(observations[i], 2)`).


For some continuous control tasks, one may consider the following notes when applying
`NaturalPolicyGradient` to new tasks and environments:

1) For two policies that both learn to complete a task satisfactorially, the larger one
    may not perform significantly better. A minimum amount of representational power is
    necessary, but larger networks may not offer quantitative benefits. The same goes for
    the value function approximator.
2) `Hmax` needs to be sufficiently long for the correct behavior to emerge; `N` needs to be
    sufficiently large that the agent samples useful data. They may also be surprisingly
    small for simple tasks. These parameters are the main tunables when applying `NaturalPolicyGradient`.
3) One might consider the `norm_step_size` and `max_cg_iter` parameters as the next most
    important when initially testing `NaturalPolicyGradient` on new tasks, assuming `Hmax` and `N` are
    appropriately chosen for the task. `gamma` has interaction with `Hmax`,
    while the default value for `gaelambda` has been empirically found to be stable for a
    wide range of tasks.

For more details, see Algorithm 1 in
[Towards Generalization and Simplicity in Continuous Control](https://arxiv.org/pdf/1703.02660.pdf).
"""
function NaturalPolicyGradient{DT}(
    env_tconstructor,
    policy,
    value,
    valuefit!;
    Hmax::Integer = 1024,
    N::Integer = 10 * Hmax,
    Nmean::Integer = min(3*Hmax, N),
    norm_step_size::Real = 0.05,
    gamma::Real = 0.995,
    gaelambda::Real = 0.98,
    max_cg_iter::Integer = 15,
    cg_tol::Real = sqrt(eps(real(DT))),
    whiten_advantages::Bool = false,
    bootstrapped_nstep_returns::Bool = false,
    value_feature_op = nothing
) where {DT <: AbstractFloat}
    e = first(env_tconstructor(1))
    np = nparams(policy)
    z(d...) = zeros(DT, d...)

    0 < np || throw(ArgumentError("policy has no parameters"))
    0 < Hmax <= N || throw(ArgumentError("Hmax must be in interval (0, N]"))
    0 < N || throw(ArgumentError("N must be > 0"))
    0 < Nmean <= N || throw(ArgumentError("Nmean must be in interval (0, N]"))
    0 < norm_step_size || throw(ArgumentError("norm_step_size must be > 0"))
    0 < gamma <= 1 || throw(ArgumentError("gamma must be in interval (0, 1]"))
    0 < gaelambda <= 1 || throw(ArgumentError("gaelambda must be in interval (0, 1]"))
    0 < max_cg_iter || throw(ArgumentError("max_cg_iter must be > 0"))
    0 < cg_tol || throw(ArgumentError("cg_tol must be > 0"))

    envsampler = EnvSampler(env_tconstructor, dtype=DT)
    fvp_op = FVP(z(np, N), true) # `true` computes FVPs with 1/N normalization
    cg_op = CG{DT}(np, N)

    NaturalPolicyGradient(
        envsampler,
        policy,
        value,
        valuefit!,
        value_feature_op,
        Hmax,
        N,
        Nmean,
        DT(norm_step_size),
        DT(gamma),
        DT(gaelambda),
        whiten_advantages,
        bootstrapped_nstep_returns,
        max_cg_iter,
        DT(cg_tol),
        z(np),
        z(np),
        fvp_op,
        cg_op,
        z(N),
        z(N)
    )
end

function NaturalPolicyGradient(args...; kwargs...)
    NaturalPolicyGradient{Float32}(args...; kwargs...)
end


function Base.iterate(npg::NaturalPolicyGradient{DT}, i = 1) where {DT}
    @unpack envsampler, policy, value, valuefit!, Hmax, N, gamma, gaelambda = npg
    @unpack vanilla_pg, natural_pg = npg
    @unpack advantages_vec, returns_vec = npg
    @unpack fvp_op, cg_op = npg

    meanbatch = @closure sample!(randreset!, envsampler, npg.Nmean, Hmax=Hmax) do action, state, observation
        action .= policy(observation)
    end
    meanbatch = deepcopy(meanbatch) # TODO cxs

    # Perform rollouts with last policy
    elapsed_sample = @elapsed begin
        batch = @closure sample!(
            randreset!,
            envsampler,
            N,
            Hmax = Hmax,
        ) do action, state, observation
            sample!(action, policy, observation)
        end
    end

    @unpack observations, terminal_observations, actions, rewards, evaluations = batch
    obs_mat     = flatview(observations)
    termobs_mat = flatview(terminal_observations)
    act_mat     = flatview(actions)
    advantages  = batchlike(rewards, advantages_vec)
    returns     = batchlike(rewards, returns_vec)
    trajlengths = map(length, rewards)

    if npg.value_feature_op !== nothing
        feat_mat = npg.value_feature_op(observations)
        termfeat_mat = npg.value_feature_op(termobs_mat, trajlengths)
    else
        feat_mat = obs_mat
        termfeat_mat = termobs_mat
    end

    # Get baseline and terminal values for the current batch using the last value function
    baseline_vec = dropdims(value(feat_mat), dims = 1)
    baseline = batchlike(rewards, baseline_vec)
    termvals = dropdims(value(termfeat_mat), dims = 1)

    # Compute normalized GAE advantages and returns
    GAEadvantages!(advantages, baseline, rewards, termvals, gamma, gaelambda)
    npg.whiten_advantages && whiten!(advantages_vec)
    if npg.bootstrapped_nstep_returns
        bootstrapped_nstep_returns!(returns, rewards, termvals, gamma)
    else
        nstep_returns!(returns, rewards, gamma)
    end

    # Fit value function to the current batch
    elapsed_valuefit = @elapsed foreach(noop, valuefit!(value, feat_mat, returns_vec))

    # Compute ∇log π_θ(at | ot)
    elapsed_gradll = @elapsed grad_loglikelihood!(
                                                  fvp_op.glls,
                                                  policy,
                                                  act_mat,
                                                  obs_mat,
                                                 )

    # Compute the "vanilla" policy gradient as 1/T * grad_loglikelihoods * advantages_vec
    elapsed_vpg = @elapsed mul!(
                                vanilla_pg,
                                fvp_op.glls,
                                advantages_vec,
                                one(DT) / N,
                                zero(DT)
                               )

    # solve for natural_pg = FIM * vanilla_pg using conjugate gradients
    # where the full FIM is avoiding using Fisher Vector Products
    fill!(natural_pg, zero(DT))
    elapsed_cg = @elapsed cg_op(
                                natural_pg,
                                fvp_op,
                                vanilla_pg;
                                tol = npg.cg_tol,
                                maxiter = npg.max_cg_iter,
                                initiallyzero = true
                               )


    # update policy parameters: θ += alpha * natural_pg
    alpha = sqrt(npg.norm_step_size / dot(natural_pg, vanilla_pg))
    lmul!(alpha, natural_pg)
    any(isnan, natural_pg) && throw(DomainError(natural_pg, "NaN detected in gradient"))
    flatupdate!(policy, natural_pg)

    result = (
        iter = i,
        elapsed_sampled = elapsed_sample,
        elapsed_gradll = elapsed_gradll,
        elapsed_vpg = elapsed_vpg,
        elapsed_cg = elapsed_cg,
        elapsed_valuefit = elapsed_valuefit,

        meantraj_reward = mean(sum, meanbatch.rewards),
        stoctraj_reward = mean(sum, batch.rewards),

        meantraj_eval = mean(sum, meanbatch.evaluations),
        stoctraj_eval = mean(sum, batch.evaluations),

        meantraj_length = mean(length, meanbatch),
        stoctraj_length = mean(length, batch),

        meantraj_ctrlnorm = mean(traj -> mean(norm, eachcol(traj)), meanbatch.actions),
        stoctraj_ctrlnorm = mean(traj -> mean(norm, eachcol(traj)), batch.actions),

        meanterminal_reward = mean(last, meanbatch.rewards),
        stocterminal_reward = mean(last, batch.rewards),

        meanterminal_eval = mean(last, meanbatch.evaluations),
        stocterminal_eval = mean(last, batch.evaluations),

        stocbatch = deepcopy(batch),
        meanbatch = deepcopy(meanbatch),

        vpgnorm = norm(vanilla_pg),
        npgnorm = norm(natural_pg),
    )

    return result, i + 1
end
