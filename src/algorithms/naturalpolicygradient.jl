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

Construct an instance of `NaturalPolicyGradient` with `args` and `kwargs`, where
`DT <: AbstractFloat` is the element type used for pre-allocated buffers, which defaults to
Float32.

In the following explanation of the `NaturalPolicyGradient` constructor, we use the
following notation/definitions:
- `dim_o = length(observationspace(env))`
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
    - `value(obs::AbstractVector)` --> `reward::Real`, where `size(obs) == (dim_o, )`
    - `value(obs::AbstractMatrix)` --> `reward::AbstractVector`,
        where `size(obs) == (dim_o, N)` and `size(reward) == (N, )`.
- `valuefit!`: a function with signature `valuefit!(value, obs::AbstractMatrix, returns::AbstractVector)`,
    where `size(obs) == (dim_o, N)` and `size(returns) == (N, )`, that fits `value` to
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

    envsampler = EnvironmentSampler(env_tconstructor, dtype = DT)
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

    # Perform rollouts with last policy
    elapsed_sample = @elapsed begin
        batch = @closure sample(envsampler, N, reset! = randreset!, Hmax = Hmax, dtype=DT) do a, s, o
            sample!(a, policy, o)
        end
    end

    @unpack O, A, R, oT = batch
    O_mat = SpecialArrays.flatten(O)::AbstractMatrix
    A_mat = SpecialArrays.flatten(A)::AbstractMatrix
    oT_mat = SpecialArrays.flatten(oT)::AbstractMatrix
    advantages  = batchlike(advantages_vec, batch)
    returns     = batchlike(returns_vec, batch)

    if npg.value_feature_op !== nothing
        # TODO change this
        feat_mat = npg.value_feature_op(O_mat)
        termfeat_mat = npg.value_feature_op(oT_mat, map(length, batch))
    else
        feat_mat = O_mat
        termfeat_mat = oT_mat
    end

    # Get baseline and terminal values for the current batch using the last value function
    baseline_vec = dropdims(value(feat_mat), dims = 1)
    baseline = batchlike(baseline_vec, batch)
    termvals = dropdims(value(termfeat_mat), dims = 1)
    R = batchlike(R, batch)

    # Compute normalized GAE advantages and returns
    GAEadvantages!(advantages, baseline, R, termvals, gamma, gaelambda)
    npg.whiten_advantages && whiten!(advantages_vec)
    if npg.bootstrapped_nstep_returns
        bootstrapped_nstep_returns!(returns, R, termvals, gamma)
    else
        nstep_returns!(returns, R, gamma)
    end

    # Fit value function to the current batch
    elapsed_valuefit = @elapsed foreach(noop, valuefit!(value, feat_mat, returns_vec))

    # Compute ∇log π_θ(at | ot)
    elapsed_gradll = @elapsed grad_loglikelihood!(
                                                  fvp_op.glls,
                                                  policy,
                                                  A_mat,
                                                  O_mat,
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
        elapsed = (
            sampled = elapsed_sample,
            gradll = elapsed_gradll,
            vpg = elapsed_vpg,
            cg = elapsed_cg,
            valuefit = elapsed_valuefit,
        ),
        batch = batch,
        vpgnorm = norm(vanilla_pg),
        npgnorm = norm(natural_pg),
    )

    return result, i + 1
end



@inline function batchlike(A::AbsVec, B::AbsVec{<:AbsVec}) # TODO document
    offsets = Vector{Int}(undef, length(B) + 1)
    offsets[1] = 0
    for i in LinearIndices(B)
        offset = offsets[i] + length(B[i])
        checkbounds(A, offset)
        offsets[i + 1] = offset
    end
    BatchedVector(A, offsets)
end

@inline function batchlike(A::AbsVec, B::LyceumBase.TrajectoryBuffer) # TODO document
    BatchedVector(A, copy(B.offsets))
end