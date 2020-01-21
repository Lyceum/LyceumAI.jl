abstract type AbstractPolicy end

"""
    $(TYPEDEF)

`DiagGaussianPolicy` policy represents a stochastic control policy, represented as a
multivariate Gaussian distribution of the form:

```math
\\pi(a | o) \\sim \\mathcal{N}(\\mu_{\\theta_1}(o), \\Sigma_{\\theta_2})
```

where ``\\mu_{\\theta_1}`` is a neural network, parameterized by ``\\theta_1``, that maps
an observation to a mean action and ``\\Sigma_{\\theta_2}`` is a diagonal
covariance matrix parameterized by ``\\theta_2``, the diagonal entries of the matrix.
Rather than tracking ``\\Sigma_{\\theta_2}`` directly, we track the log standard deviations,
which are easier to learn. Note that ``\\mu_{\\theta_1}`` is a _state-dependent_ mean
while ``\\Sigma_{\\theta_2}`` is a _global_ covariance.
"""
struct DiagGaussianPolicy{Mean,Logstd<:AbstractVector} <: AbstractPolicy
    meanNN::Mean
    logstd::Logstd
    fixedlogstd::Bool
end

"""
    $(TYPEDSIGNATURES)

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


function sample!(rng::AbstractRNG, actions::AbsVec, policy::DiagGaussianPolicy, features::AbsVec)
    randn!(rng, actions)
    actions .= actions .* exp.(policy.logstd) .+ policy(features)
end
function sample!(actions::AbsVec, policy::DiagGaussianPolicy, features::AbsVec)
    sample!(default_rng(), actions, policy, features)
end


(policy::DiagGaussianPolicy)(features) = policy.meanNN(features)

function getaction!(actions::AbsVec, policy::DiagGaussianPolicy, features::AbsVec)
    actions .= actions .* exp.(policy.logstd) .+ policy(features)
end


function loglikelihood(P::DiagGaussianPolicy, feature::AbsVec, action::AbsVec)
    meanact = P(feature)
    ll = -length(P.logstd) * log(2pi) / 2
    for i = 1:length(action)
        ll -= ((meanact[i] - action[i]) / exp(P.logstd[i]))^2 / 2
        ll -= P.logstd[i]
    end
    ll
end

function loglikelihood(m::DiagGaussianPolicy, feats::AbsMat, acts::AbsMat)
    constterm = -length(m.logstd) * log(2pi) / 2 - sum(m.logstd)
    meanacts = m(feats)

    #zs2 = ((meanacts .- acts) ./ exp.(m.logstd)) .^ 2
    # NOTE: this is about 2x faster than the above line
    zs = ((meanacts .- acts) ./ exp.(m.logstd))
    zs = zs .^ 2

    #ll = DT(-0.5) * sum(zs, dims=1) .+ constterm
    ll = constterm .- sum(zs, dims = 1) / 2
    dropdims(ll, dims = 1)
end


function grad_loglikelihood!(
    gradll::AbsVec,
    m::DiagGaussianPolicy,
    feat::AbsVec,
    act::AbsVec,
    ps = params(m),
)
    gs = gradient(() -> loglikelihood(m, feat, act), params(m))
    copygrad!(gradll, ps, gs)
    gradll
end

@propagate_inbounds function grad_loglikelihood!(
    gradlls::AbsMat,
    policy::DiagGaussianPolicy,
    features::AbsMat,
    actions::AbsMat,
    nthreads::Int = Threads.nthreads(),
)
    @boundscheck begin
        if !(axes(gradlls, 2) == axes(features, 2) == axes(actions, 2))
            throw(ArgumentError("gradlls, features, and actions must have same 2nd axis"))
        end
        require_one_based_indexing(gradlls, features, actions)
    end

    bthreads = max(1, div(Threads.nthreads(), nthreads))
    ranges = splitrange(size(gradlls, 2), nthreads)
    @with_blasthreads bthreads begin
        @sync for r in ranges
            Threads.@spawn _threaded_gradll!(gradlls, policy, features, actions, r)
        end
    end
    gradlls
end

function _threaded_gradll!(
    gradlls::AbsMat,
    m::DiagGaussianPolicy,
    feats::AbsMat,
    acts::AbsMat,
    range::UnitRange{Int},
)
    for i in range
        feat = uview(feats, :, i)
        act = uview(acts, :, i)
        gradll = uview(gradlls, :, i)
        grad_loglikelihood!(gradll, m, feat, act)
    end
    gradlls
end

