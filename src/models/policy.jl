"""
    $(TYPEDEF)

`DiagGaussianPolicy` policy represents a stochastic control policy, represented as a
multivariate Gaussian distribution of the form:

```math
\\pi_{\\theta}(a | o) = \\mathcal{N}(\\mu_{\\theta_1}(o), \\Sigma_{\\theta_2})
```

where ``\\mu_{\\theta_1}`` is a neural network, parameterized by ``\\theta_1``, that maps
an observation to a mean action and ``\\Sigma_{\\theta_2}`` is a diagonal
covariance matrix parameterized by ``\\theta_2``, the diagonal entries of the matrix.
Rather than tracking ``\\Sigma_{\\theta_2}`` directly, we track the log standard deviations,
which are easier to learn. Note that ``\\mu_{\\theta_1}`` is a _state-dependent_ mean
while ``\\Sigma_{\\theta_2}`` is a _global_ covariance.
"""
struct DiagGaussianPolicy{Mean,Logstd<:AbstractVector}
    meanNN::Mean
    logstd::Logstd
    fixedlogstd::Bool
end

"""
    $(SIGNATURES)

Construct a `DiagGaussianPolicy` with a state-dependent mean `meanNN` and initial
log-standard deviation `logstd`. If `fixedlogstd` is true, `logstd` will be treated as
a constant. `meanNN` should be object that is compatible with Flux.jl and have the
following signatures:
- `meanNN(obs::AbstractVector)` --> `action::AbstractVector`
- `meanNN(obs::AbstractMatrix)` --> `action::AbstractMatrix`
"""
function DiagGaussianPolicy(meanNN, logstd::AbstractVector; fixedlogstd::Bool = false)
    DiagGaussianPolicy(meanNN, logstd, fixedlogstd)
end


Flux.@functor DiagGaussianPolicy

function Flux.trainable(policy::DiagGaussianPolicy)
    if policy.fixedlogstd
        (meanNN = policy.meanNN, ) \propagate_inbounds
    else
        (meanNN = policy.meanNN, logstd = policy.logstd)
    end
end

"""
    sample!([rng = GLOBAL_RNG, ]action, policy, feature)

Treating `policy` as a stochastic policy, sample an action from `policy`, conditioned on
`feature`, and store it in `action`.
"""
function sample!(rng::AbstractRNG, action::AbsVec, policy::DiagGaussianPolicy, feature::AbsVec)
    randn!(rng, action)
    action .= action .* exp.(policy.logstd) .+ policy(feature)
end
function sample!(action::AbsVec, policy::DiagGaussianPolicy, feature::AbsVec)
    sample!(default_rng(), action, policy, feature)
end


(policy::DiagGaussianPolicy)(features) = policy.meanNN(features)

"""
    $(SIGNATURES)

Treating `policy` as a deterministic policy, compute the mean action of `policy`,
conditioned on `feature`, and store it in `action`.
"""
function getaction!(action::AbsVec, policy::DiagGaussianPolicy, feature::AbsVec)
    action .= action .* exp.(policy.logstd) .+ policy(feature)
end

"""
    $(SIGNATURES)

Return loglikelihood of `action` conditioned on `feature` for `policy`.
"""
function loglikelihood(policy::DiagGaussianPolicy, action::AbsVec, feature::AbsVec)
    meanact = policy(feature)
    ls = policy.logstd
    ll = -length(ls) * log(2pi) / 2
    for i = 1:length(action)
        ll -= ((meanact[i] - action[i]) / exp(ls[i]))^2 / 2
        ll -= ls[i]
    end
    ll
end

"""
    $(SIGNATURES)

Treating each column of `actions` and `features` as a single action/feature, return
a vector of the loglikelihoods of `actions` conditioned on `features` for `policy`.
"""
function loglikelihood(policy::DiagGaussianPolicy, actions::AbsMat, features::AbsMat)
    constterm = -length(policy.logstd) * log(2pi) / 2 - sum(policy.logstd)
    meanactions = policy(features)

    #zs2 = ((meanactions .- actions) ./ exp.(m.logstd)) .^ 2
    # NOTE: this is about 2x faster than the above line
    zs = ((meanactions .- actions) ./ exp.(policy.logstd))
    zs = zs .^ 2

    #ll = DT(-0.5) * sum(zs, dims=1) .+ constterm
    ll = constterm .- sum(zs, dims = 1) / 2
    dropdims(ll, dims = 1)
end


"""
    $(SIGNATURES)

Return the gradient of the loglikelihood of `action` conditioned on `feature` with respect
to `policy`'s parameters.
"""
function grad_loglikelihood!(
    gradll::AbsVec,
    policy::DiagGaussianPolicy,
    action::AbsVec,
    feature::AbsVec
)
    ps = params(policy)
    gs = gradient(() -> loglikelihood(policy, action, feature), ps)
    copygrad!(gradll, ps, gs)
    gradll
end

"""
    $(SIGNATURES)

Treating each column of `actions` and `features` as a single action/feature,
return a vector of gradients of the loglikelihood of `action` conditioned on `feature` with respect
to `policy`'s parameters.

This computation is done in parallel using `nthreads` threads.
"""
@propagate_inbounds function grad_loglikelihood!(
    gradlls::AbsMat,
    policy::DiagGaussianPolicy,
    actions::AbsMat,
    features::AbsMat,
    nthreads::Int = Threads.nthreads(),
)
    @boundscheck begin
        if !(axes(gradlls, 2) == axes(actions, 2) == axes(features, 2))
            throw(ArgumentError("gradlls, actions, and features must have same 2nd axis"))
        end
        require_one_based_indexing(gradlls, actions, features)
    end

    bthreads = max(1, div(Threads.nthreads(), nthreads))
    ranges = splitrange(size(gradlls, 2), nthreads)
    @with_blasthreads bthreads begin
        @sync for r in ranges
            Threads.@spawn _threaded_gradll!(gradlls, policy, actions, features, r)
        end
    end
    gradlls
end

function _threaded_gradll!(
    gradlls::AbsMat,
    m::DiagGaussianPolicy,
    acts::AbsMat,
    feats::AbsMat,
    range::UnitRange{Int},
)
    for i in range
        feat = view(feats, :, i)
        act = view(acts, :, i)
        gradll = view(gradlls, :, i)
        grad_loglikelihood!(gradll, m, act, feat)
    end
    gradlls
end

