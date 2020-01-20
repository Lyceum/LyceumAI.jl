struct NaturalPolicyGradient{DT,S,P,V,VF,VOP,CB}
    envsampler::S
    policy::P

    value::V
    valuefit!::VF
    value_feature_op::VOP

    Hmax::Int
    N::Int
    Nmean::Int
    stopcb::CB

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

    function NaturalPolicyGradient(
        env_tconstructor,
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
        max_cg_iter = 12, # maybe 2 * action dim is a better heuristic?
        cg_tol = 1e-18, # prob something like 1e-9 or 1e-6...
        whiten_advantages = false,
        bootstrapped_nstep_returns = false,
        value_feature_op = nothing
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

        envsampler = EnvSampler(env_tconstructor, dtype=DT)

        z(d...) = zeros(DT, d...)
        fvp_op = FVP(z(np, N), true) # `true` computes FVPs with 1/N normalization
        cg_op = CG{DT}(np, N)

        new{
            DT,
            typeof(envsampler),
            typeof(policy),
            typeof(value),
            typeof(valuefit!),
            typeof(value_feature_op),
            typeof(stopcb)
        }(
            envsampler,
            policy,
            value,
            valuefit!,
            value_feature_op,
            Hmax,
            N,
            Nmean,
            stopcb,
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
end

# Algorithm 1 in "Towards Generalization and Simplicity in Continuous Control"
# https://arxiv.org/pdf/1703.02660.pdf
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
            randn!(action) # TODO noise buffer for better determinism
            getaction!(action, policy, observation)
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
                                                  obs_mat,
                                                  act_mat,
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
    # TODO better initial guess?
    natural_pg .= zero(eltype(natural_pg)) # start CG from initial guess of 0
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
