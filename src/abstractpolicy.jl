abstract type AbstractPolicy end
abstract type AbstractDeterministicPolicy <: AbstractPolicy end
abstract type AbstractStochasticPolicy <: AbstractPolicy end

# user implements:
# Obs/Act space policy?
# condition
# optionally:
# _mean(p, V)
# _mean(p, M)
# _logpdf(p, V, V)
# _logpdf(p, M, M)
# _pdf(p, V, V)
# _pdf(p, M, M)

@mustimplement Statistics.mean(π::AbstractStochasticPolicy, o::AbsVec)
function Statistics.mean(π::AbstractStochasticPolicy, O::AbsMat)
    @views mapreduce(hcat, axes(O, 2)) do i
        @_inline_meta
        @inbounds mean(π, O[:,i])
    end
end

Statistics.mean!(a::AbsVec, π::AbstractStochasticPolicy, o::AbsVec) = copyto!(a, mean(π, o))
Statistics.mean!(A::AbsMat, π::AbstractStochasticPolicy, O::AbsMat) = copyto!(A, mean(π, O))


@mustimplement Distributions.logpdf(π::AbstractStochasticPolicy, o::AbsVec, a::AbsVec)
function Distributions.logpdf(π::AbstractStochasticPolicy, O::AbsMat, A::AbsMat)
    @views map(eachindex(axes(O, 2), axes(A, 2))) do i
        @_inline_meta
        @inbounds logpdf(π, O[:, i], A[:, i])
    end
end

function Distributions.logpdf!(r::AbsVec, π::AbstractStochasticPolicy, O::AbsMat, A::AbsMat)
    copyto!(r, logpdf(π, O, A))
    return r
end


Distributions.pdf(π::AbstractStochasticPolicy, o::AbsVec, a::AbsVec) = exp(logpdf(π, o, a))
Distributions.pdf(π::AbstractStochasticPolicy, O::AbsMat, A::AbsMat) = exp.(logpdf(π, O, A))

function Distributions.pdf!(r::AbsVec, π::AbstractStochasticPolicy, O::AbsMat, A::AbsMat)
    copyto!(r, pdf(π, O, A))
    return r
end


Random.rand(rng::AbstractRNG, π::AbstractStochasticPolicy, o::AbsVec) = rand(rng, condition(π, o))
function Random.rand(rng::AbstractRNG, π::AbstractStochasticPolicy, O::AbsMat)
    @views mapreduce(hcat, eachindex(axes(O, 2))) do i
        @_inline_meta
        @inbounds rand(rng, π, O[:,i])
    end
end
Random.rand(π::AbstractStochasticPolicy, o::AbsVec) = rand(default_rng(), π, o)
Random.rand(π::AbstractStochasticPolicy, O::AbsMat) = rand(default_rng(), π, O)

function Random.rand!(rng::AbstractRNG, π::AbstractStochasticPolicy, o::AbsVec, a::AbsVec)
    copyto!(A, rand(rng, π, o))
end
function Random.rand!(rng::AbstractRNG, π::AbstractStochasticPolicy, O::AbsMat, A::AbsMat)
    copyto!(A, rand(rng, π, O))
end
Random.rand!(π::AbstractStochasticPolicy, o::AbsVec, a::AbsVec) = rand!(default_rng(), π, o, a)
Random.rand!(π::AbstractStochasticPolicy, O::AbsMat, A::AbsMat) = rand!(default_rng(), π, O, A)
