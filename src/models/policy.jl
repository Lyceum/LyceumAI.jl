abstract type AbstractPolicy end


struct DiagGaussianPolicy{F,M,L<:AbstractVector} <: AbstractPolicy
    meanNN::M
    logstd::L
    function DiagGaussianPolicy{F,M,L}(meanNN, logstd) where {F,M,L<:AbstractVector}
        F isa Bool || throw(ArgumentError("Type parameter F must be <: Bool"))
        new{F,M,L}(meanNN, logstd)
    end
end

function DiagGaussianPolicy{F}(meanNN, logstd) where {F}
    DiagGaussianPolicy{F,typeof(meanNN),typeof(logstd)}(meanNN, logstd)
end

"""
    DiagGaussianPolicy( neuralnetwork, ::Vector{T} )

This struct presents a Policy object for policy gradient methods. It contains a function approximator object,
often a neural netowrk and a vector of values representing the standard deviation of noise to make the policy
stochastic. When the policy is sampled, this noise is exponentated before added to the network output. See 
`getaction!(actions::AbsVec, policy::DiagGaussianPolicy, features::AbsVec)`for more details.

```julia-repl
julia> policy = DiagGaussianPolicy(
           multilayer_perceptron(
               dobs,
               32,
               32,
               dact;
               σ = tanh,
               initb = glorot_uniform,
               initb_final = glorot_uniform,
           ),
           zeros(dact)
       )
```
"""
DiagGaussianPolicy(meanNN, logstd) = DiagGaussianPolicy{true}(meanNN, logstd)


Flux.@functor DiagGaussianPolicy
Flux.trainable(policy::DiagGaussianPolicy{true}) =
    (meanNN = policy.meanNN, logstd = policy.logstd)
Flux.trainable(policy::DiagGaussianPolicy{false}) = (meanNN = policy.meanNN,)


(policy::DiagGaussianPolicy)(features) = policy.meanNN(features)

"""
    getaction!(actions::AbstractVector, policy::DiagGaussianPolicy, features::AbstractVector)

Treats the policy object as a stochastic Gaussian policy and samples from it conditioned on `features`:

    π( actions | features )

The policy object stores the standard deviation of the policy distribution in log form for numerical stability.
The evaluation of the policy represents the mean of the normal. This function assumes `action` has been populated with N(0, 1) unit normal random samples, which is scaled by the policy's standard deviation and added to it's mean.
"""
function getaction!(actions::AbsVec, policy::DiagGaussianPolicy, features::AbsVec)
    actions .= actions .* exp.(policy.logstd) .+ policy(features)
end

"""
    loglikelihood(P::DiagGaussianPolicy, feature::AbstractVector, action::AbstractVector)

The loglikelihood of the stochastic policy; calculated for single inputs.

"""
function loglikelihood(P::DiagGaussianPolicy, feature::AbsVec, action::AbsVec)
    meanact = P(feature)
    ll = -length(P.logstd) * log(2pi) / 2
    for i = 1:length(action)
        ll -= ((meanact[i] - action[i]) / exp(P.logstd[i]))^2 / 2
        ll -= P.logstd[i]
    end
    ll
end

"""
    loglikelihood(P::DiagGaussianPolicy, feature::AbstractMatrix, action::AbstractMatrix)

The loglikelihood of the stochastic policy; calculated for multiple inputs in matrix form.
"""
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

"""
    grad_loglikelihood!(gradll::AbstractVector, m::DiagGaussianPolicy, feat::AbstractVector, act::AbstractVector, ps = params(m))

Calculate the gradient of the policy's log-likelihood for a given singular input. This function in-place writes to an allocated vector `gradll` of length equal to the number of parameters of the policy, including the logstd vector.
"""
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

"""
    grad_loglikelihood!(gradll::AbstractMatrix, m::DiagGaussianPolicy, feat::AbstractMatrix, act::AbstractMatrix, ps = params(m))

Calculate the gradient of the policy's log-likelihood for input batch. This function in-place writes to an allocated vector `gradll` of length equal to the number of parameters of the policy, including the logstd vector.

This function dispatches multiple threads to evaluate the gradient function. If you have signaled the julia to use mutliple threads, you may experience a speedup.
"""
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
# TODO need to zero out w/ Zyogote?

#function _gradllkernel(data, ms, r)
#    pol = trainmode(getmine(ms))
#    for i in r
#        gll, f, a = data[i]
#        grad_loglikelihood!(gll, pol, f, a)
#    end
#    nothing
#end
#
#function grad_loglikelihood!(gradlls::AbsMat, ms::ThreadedFluxModel{<:DiagGaussianPolicy},
#    feats::AbsMat, acts::AbsMat;
#    nthreads=div(size(gradlls, 2), 1024))
#
#    N = size(gradlls, 2)
#    jthreads = Threads.nthreads()
#    nthreads = max(1, min(N, nthreads))
#    nthreads = min(jthreads, nthreads)
#    bthreads = max(1, div(jthreads, nthreads))
#    oldbthreads = nblasthreads()
#
#    ranges = Distributed.splitrange(N, nthreads)
#    data = eachsample(gradlls, feats, acts)
#    BLAS.set_num_threads(bthreads)
#    #GC.enable(false)
#    @sync for tid=1:nthreads
#        Threads.@spawn _gradllkernel(data, ms, ranges[tid])
#    end
#    #GC.enable(true); GC.gc()
#    BLAS.set_num_threads(oldbthreads)
#    gradlls
#end
