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

    @doc """

        NaturalPolicyGradient( env_tconstructor, policy, value, valuefit!)

    Constructor for NPG algorithm struct, with parameters below:

        env_tconstructor = expects function of an int i, returns i instances of environment
        policy           = policy function that accepts observations, [ obssize x n ], and returns [ ctrlsize x n ] action vectors. See LyceumAI.jl/src/models/policy.jl for policy structs.
        value            = similar to policy but represents the estimated value for a given observation [ obssize x n ] and returns [ 1 x n ] values.
        valuefit!        = function with signature valuefit!(value, obs_mat, returns_vec) that fits the above value function to data; see LyceumAI.jl/src/flux.jl for FluxTrainer example.

        Hmax                       = Maximum Length of a sample Trajectory
        N                          = Total number of data samples used for each policy gradient step
        Nmean                      = Total number of data samples for the mean policy (without stochastic noise)
        stopcb                     = A callback to signal when to break out of iteration
        norm_step_size             = Scaling for weighing the applied gradient update after gradient normalization has occured. This process makes training much more stable to step sizes; see equation 5 in https://arxiv.org/pdf/1703.02660.pdf for more details
        gamma                      = Reward discount, applied as `for t = 1:Hmax ( gamma^(t-1) * reward[t] ) end`
        gaelambda                  = Generalized Advantage Estimate parameter, balances bias and variance comparing current reward with value function estimate. See https://arxiv.org/pdf/1506.02438.pdf for details.
        max_cg_iter                = Conjugate Gradient is used to estimate the inverse Fisher * gradient computation; this is the number of iterations to use
        cg_tol                     = The numerical tolerance used to break out of the CG loop
        whiten_advantages          = Boolean flag for whether to apply statistical whitening to calculated advantages (resulting in zero mean, 1 stdev data)
        bootstrapped_nstep_returns = Boolean flag controls the return calculation for the terminal state of a rollout trajectory


    For some continuous control tasks, one may consider the following notes and "anecdata" when applying NPG to new tasks and environments:

    1) For two policies that both learn to complete a task satisfactorially, the larger one may not perform significantly better. A minimum amount of representational power is necessary, but larger networks may not offer quantitative benefits. The same goes for the value function approximator.
    2) Hmax needs to be sufficiently long for the correct behavior to emerge; N needs to be sufficiently large that the agent samples useful data. They may also be surprisingly small for simple tasks. These parameters are the main tunables when applying NPG.
    3) One might consider the norm_step_size and max_cg_iter paramters as the next most important when initially testing NPG on new tasks, assuming Hmax and N are appropriately chosen for the task. gamma has interaction with Hmax, while gaelambda's default has been empirically found to be stable for a wide range of tasks.

    # Example that assumes policy and value functions have been constructed
    ```julia-repl
    julia> npg = NaturalPolicyGradient(
              (i) -> tconstructor(env, i),
              policy,
              value,
              valuefit!)
    ```

    See for more details:
    Algorithm 1 in "Towards Generalization and Simplicity in Continuous Control"
    https://arxiv.org/pdf/1703.02660.pdf
    """
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

"""
    Base.iterate(npg::NaturalPolicyGradient{DT}, i = 1) where {DT}

Iterator function for the NaturalPolicyGradient struct.

# Example that assumes policy and value functions have been constructed
```julia-repl
julia> npg = NaturalPolicyGradient(
        (i) -> tconstructor(env, i),
        policy,
        value,
        valuefit!)
julia> for (i, state) in enumerate(npg)
          if i >= 200
             break # Iterates the NPG algorithm 200 times.
          end
       end

julia> state, i = iterate(npg, 1) # runs one step of the algorithm, returning state
```

See for more details:
Algorithm 1 in "Towards Generalization and Simplicity in Continuous Control"
https://arxiv.org/pdf/1703.02660.pdf
"""
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
