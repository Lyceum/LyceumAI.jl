struct NaturalPolicyGradient{DT,S,P,V,VF,CB}
    envsampler::S
    policy::P

    value::V
    valuefit!::VF

    Hmax::Int
    N::Int
    Nmean::Int
    stopcb::CB

    norm_step_size::DT
    gamma::DT
    gaelambda::DT

    max_cg_iter::Int
    cg_tol::DT

    vanilla_pg::Vector{DT} # nparams
    natural_pg::Vector{DT} # nparams

    grad_loglikelihoods::Array{DT,2} # nparams x N
    fisher_information_matrix::Symmetric{DT} # nparams x nparams

    advantages_vec::Vector{DT} # N
    returns_vec::Vector{DT} # N

    function NaturalPolicyGradient(
        env_ctor,
        policy,
        value,
        valuefit!;
        Hmax = 100,
        N = 5120,
        Nmean = max(Hmax*2, N),
        stopcb = infinitecb,
        norm_step_size = 0.05,
        gamma = 0.995,
        gaelambda = 0.98,
        max_cg_iter = 12,
        cg_tol = 1e-18,
    )


        0 < Hmax <= N || throw(ArgumentError("Hmax must be in interval (0, N]"))
        0 < N || throw(ArgumentError("N must be > 0"))
        0 < Nmean <= N || throw(ArgumentError("Nmean must be in interval (0, N]"))
        0 < norm_step_size || throw(ArgumentError("norm_step_size must be > 0"))
        0 < gamma <= 1 || throw(ArgumentError("gamma must be in interval (0, 1]"))
        0 < gaelambda <= 1 || throw(ArgumentError("gaelambda must be in interval (0, 1]"))
        0 < max_cg_iter || throw(ArgumentError("max_cg_iter must be > 0"))
        0 < cg_tol || throw(ArgumentError("cg_tol must be > 0"))
        hasmethod(
            valuefit!,
            (typeof(value), Matrix, Vector),
        ) || throw(ArgumentError("valuefit! must have signature: (value, Matrix, Vector)"))
        hasmethod(
            stopcb,
            (NamedTuple,),
        ) || throw(ArgumentError("stopcb must have signature: (NamedTuple)"))

        np = nparams(policy)
        0 < np || throw(ArgumentError("policy has no parameters"))

        DT = promote_modeltype(policy, value)
        if !isconcretetype(DT)
            DTnew = Shapes.default_datatype(DT)
            @warn "Could not infer model element type. Defaulting to $DTnew"
            DT = DTnew
        end

        envsampler = EnvSampler(env_ctor, dtype=DT)

        z(d...) = zeros(DT, d...)
        new{
            DT,
            typeof(envsampler),
            typeof(policy),
            typeof(value),
            typeof(valuefit!),
            typeof(stopcb),
        }(
            envsampler,
            policy,
            value,
            valuefit!,
            Hmax,
            N,
            Nmean,
            stopcb,
            DT(norm_step_size),
            DT(gamma),
            DT(gaelambda),
            max_cg_iter,
            DT(cg_tol),
            z(np),
            z(np),
            z(np, N),
            Symmetric(z(np, np)),
            z(N),
            z(N),
        )
    end
end


# Algorithm 1 in "Towards Generalization and Simplicity in Continuous Control"
# https://arxiv.org/pdf/1703.02660.pdf
function Base.iterate(npg::NaturalPolicyGradient{DT}, i = 1) where {DT}
    @unpack envsampler, policy, value, valuefit!, Hmax, N, gamma, gaelambda = npg
    @unpack vanilla_pg, natural_pg, grad_loglikelihoods, fisher_information_matrix = npg
    @unpack advantages_vec, returns_vec = npg

    meanbatch = @closure sample!(reset!, envsampler, npg.Nmean, Hmax=Hmax) do action, state, observation
        action .= policy(observation)
    end
    meanbatch = deepcopy(meanbatch)

    # Perform rollouts with last policy
    elapsed_sample = @elapsed begin
        batch = @closure sample!(
            randreset!,
            envsampler,
            N,
            Hmax = Hmax,
        ) do action, state, observation
            randn!(action) # TODO noise buffer for better determinism
            getaction!(action, policy, observation)
        end
    end

    @unpack observations, terminal_observations, actions, rewards, evaluations = batch
    obs_mat = flatview(observations)
    termobs_mat = flatview(terminal_observations)
    act_mat = flatview(actions)
    advantages = batchlike(rewards, advantages_vec)
    returns = batchlike(rewards, returns_vec)

    # Get baseline and terminal values for the current batch using the last value function
    baseline_vec = dropdims(value(obs_mat), dims = 1)
    baseline = batchlike(rewards, baseline_vec)
    termvals = dropdims(value(termobs_mat), dims = 1)

    # Compute normalized GAE advantages and returns
    GAEadvantages!(advantages, baseline, rewards, termvals, gamma, gaelambda)
    #whiten!(advantages_vec)
    nstep_returns!(returns, rewards, gamma)

    # Fit value function to the current batch
    elapsed_valuefit = @elapsed foreach(noop, valuefit!(value, obs_mat, returns_vec))

    # Compute sample-wise gradient of the loglikelihoods of actions taken during the rollout
    elapsed_gradll = @elapsed grad_loglikelihood!(
        grad_loglikelihoods,
        policy,
        obs_mat,
        act_mat,
        8,
    )

    # Compute the vanilla policy gradient
    # vanilla_pg <-- 1/N * grad_loglikelihoods * advantages_vec
    elapsed_vpg = @elapsed mul!(
        vanilla_pg,
        grad_loglikelihoods,
        advantages_vec,
        one(DT) / N,
        zero(DT),
    )

    # compute Fisher matrix (ALG7)
    # fisher_information_matrix <-- 1/N * grad_loglikelihoods * transpose(grad_loglikelihoods)
    # NOTE: fisher_information_matrix isa LinearAlgebra.Symmetric, only the upper triangular portion is computed
    #elapsed_fim = @elapsed symmul!(fisher_information_matrix, grad_loglikelihoods, transpose(grad_loglikelihoods), one(DT) / N, zero(DT))
    elapsed_fim = @elapsed BLAS.syrk!(
        'U',
        'N',
        one(DT) / N,
        grad_loglikelihoods,
        zero(DT),
        fisher_information_matrix.data,
    )


    ## gradient ascent (ALG7) --
    elapsed_cg = @elapsed cg!(
        natural_pg,
        fisher_information_matrix,
        vanilla_pg;
        tol = npg.cg_tol,
        maxiter = npg.max_cg_iter,
    )

    alpha = sqrt(npg.norm_step_size / dot(natural_pg, vanilla_pg))
    lmul!(alpha, natural_pg)
    any(isnan, natural_pg) && throw(DomainError(natural_pg, "NaN detected in gradient"))

    flatupdate!(policy, natural_pg)



    result = (
        iter = i,
        elapsed_sampled = elapsed_sample,
        elapsed_gradll = elapsed_gradll,
        elapsed_vpg = elapsed_vpg,
        elapsed_fim = elapsed_fim,
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
