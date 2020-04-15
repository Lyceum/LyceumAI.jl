module M

using Base: @_inline_meta
using Flux
using Flux: @functor, params, outdims, update!
using Flux.Zygote: Params, Grads
using DistributionsAD

using Distributions: Distributions, MvNormal, pdf, pdf!, logpdf, logpdf!
using LyceumCore
using Random: Random, AbstractRNG, default_rng
using LyceumCore
using MacroTools: @forward
using RecursiveArrayTools
using Statistics: Statistics, mean, var, cov, cor
using StatsBase: StatsBase, cor, entropy



include("FluxTools.jl")

const AbsVecOrMat{T} = Union{AbstractVector{T},AbstractMatrix{T}}

####
#### Model Interface
####

#abstract type AbstractModel end
#@mustimplement (m::AbstractModel)(xs...)
#@mustimplement Flux.params(m::AbstractModel)
#@mustimplement Flux.functor(m::AbstractModel)
# (m)(xs...)
# Flux.params(m)
# Flux.functor(m)

# optional overrides
paramvec(m) = paramvec(params(m))
paramvecview(m) = paramvecview(params(m))
parameltype(m) = parameltype(params(m))
paramlength(m) = paramlength(params(m))

copyparams!(ps::ParamTypes, m) = copyparams!(ps, params(m))
copyparams!(m, ps::ParamTypes) = copyparams!(params(m), ps)

updateparams!(m, gs::GradTypes) = updateparams!(params(m), gs)

#abstract type AbstractModel end


####
#### Policy
####

abstract type AbstractPolicy end

#@mustimplement (π::AbstractPolicy)(xs...)
#@mustimplement Flux.params(π::AbstractPolicy)
#@mustimplement Flux.functor(π::AbstractPolicy)


abstract type AbstractDeterministicPolicy <: AbstractPolicy end


abstract type AbstractStochasticPolicy <: AbstractPolicy end

@mustimplement condition(π::AbstractStochasticPolicy, o::AbsVec)::Distributions.MultivariateDistribution

for f in (:var, :cov, :cor)
    @eval Statistics.$f(π::AbstractStochasticPolicy, o::AbsVec) = Statistics.$f($condition(π, o))
end

StatsBase.entropy(π::AbstractStochasticPolicy, o::AbsVec) = entropy(condition(π, o))
StatsBase.entropy(π::AbstractStochasticPolicy, o::AbsVec, b::Real) = entropy(condition(π, o), b)

function Distributions.insupport(π::AbstractStochasticPolicy, o::AbsVec, a::AbsVec)
    insupport(condition(π, o), a)
end


Statistics.mean(π::AbstractStochasticPolicy, o::AbsVec) = mean(condition(π, o))

function Statistics.mean(π::AbstractStochasticPolicy, O::AbsMat)
    A = map(axes(O, 2)) do i
        @_inline_meta
        @inbounds mean(π, view(O, :, i))
    end
    return reduce(hcat, A)
end

function Statistics.mean!(A::AbsMat, π::AbstractStochasticPolicy, O::AbsMat)
    for i in eachindex(axes(A, 2), axes(O, 2))
        @inbounds A[:, i] = mean(condition(π, view(O, :, i)))
    end
    return A
end


Distributions.pdf(π::AbstractStochasticPolicy, o::AbsVec, a::AbsVec) = pdf(condition(π, o), a)

function Distributions.pdf(π::AbstractStochasticPolicy, O::AbsMat, A::AbsMat)
    r = map(eachindex(axes(O, 2), axes(A, 2))) do i
        @_inline_meta
        @inbounds pdf(π, view(O, :, i), view(A, :, i))
    end
    return r
end

function Distributions.pdf!(r::AbsVec, π::AbstractStochasticPolicy, O::AbsMat, A::AbsMat)
    for i in eachindex(r, axes(O, 2), axes(A, 2))
        @inbounds r[i] = pdf(π, view(O, :, i), view(A, :, i))
    end
    return r
end


Distributions.logpdf(π::AbstractStochasticPolicy, o::AbsVec, a::AbsVec) = logpdf(condition(π, o), a)

function Distributions.logpdf(π::AbstractStochasticPolicy, O::AbsMat, A::AbsMat)
    r = map(eachindex(axes(O, 2), axes(A, 2))) do i
        @_inline_meta
        @inbounds logpdf(π, view(O, :, i), view(A, :, i))
    end
    return r
end

function Distributions.logpdf!(r::AbsVec, π::AbstractStochasticPolicy, O::AbsMat, A::AbsMat)
    for i in eachindex(r, axes(O, 2), axes(A, 2))
        @inbounds r[i] = logpdf(π, view(O, :, i), view(A, :, i))
    end
    return r
end


Random.rand(rng::AbstractRNG, π::AbstractStochasticPolicy, o::AbsVec) = rand(rng, condition(π, o))
Random.rand(π::AbstractStochasticPolicy, o::AbsVec) = rand(default_rng(), condition(π, o))

function Random.rand(rng::AbstractRNG, π::AbstractStochasticPolicy, O::AbsMat)
    A = map(axes(O, 2)) do i
        @_inline_meta
        @inbounds rand(rng, π, view(O, :, i))
    end
    return reduce(hcat, A)
end
Random.rand(π::AbstractStochasticPolicy, O::AbsMat) = rand(default_rng(), π, O)

function Random.rand!(rng::AbstractRNG, A::AbsMat, π::AbstractStochasticPolicy, O::AbsMat)
    for i in eachindex(axes(A, 2), axes(O, 2))
        A[:, i] = rand(rng, π, view(O, :, i))
    end
    return A
end
Random.rand!(A::AbsMat, π::AbstractStochasticPolicy, O::AbsMat) = rand!(default_rng(), A, π, O)


####
#### DiagonalGaussianPolicy
####

struct DiagonalGaussianPolicy{Mean,LogStd} <: AbstractStochasticPolicy
    mean::Mean
    logstd::LogStd
end

const FixedDiagonalGaussianPolicy = DiagonalGaussianPolicy{<:Any,<:AbsVec}

Flux.@functor DiagonalGaussianPolicy

condition(π::DiagonalGaussianPolicy, o::AbsVec) = MvNormal(π(o), exp.(π.logstd(o)))
condition(π::FixedDiagonalGaussianPolicy, o::AbsVec) = MvNormal(π(o), exp.(π.logstd))

(π::DiagonalGaussianPolicy)(o::AbsVecOrMat) = mean(π, o)

Statistics.mean(π::DiagonalGaussianPolicy, o::AbsVec) = π.mean(o)
Statistics.mean(π::DiagonalGaussianPolicy, O::AbsMat) = π.mean(O)


function Distributions.logpdf(π::DiagonalGaussianPolicy, a::AbsVec, o::AbsVec)
    _mvnormal_logpdf(mean(π, o), π.logstd(o), a)
end
function Distributions.logpdf(π::DiagonalGaussianPolicy, A::AbsMat, O::AbsMat)
    _mvnormal_logpdf(mean(π, O), π.logstd(O), A)
end

function Distributions.logpdf(π::FixedDiagonalGaussianPolicy, a::AbsVec, o::AbsVec)
    _mvnormal_logpdf(mean(π, o), π.logstd, a)
end
function Distributions.logpdf(π::FixedDiagonalGaussianPolicy, A::AbsMat, O::AbsMat)
    _mvnormal_logpdf(mean(π, O), π.logstd, A)
end

function _mvnormal_logpdf(μ::AbsVec, ls::AbsVec, x::AbsVec)
    return -(length(x) * log(2π) + 2 * sum(ls) + sum(abs2.((x .- μ) ./ exp.(ls)))) / 2
end

function _mvnormal_logpdf(μ::AbsMat, ls::AbsVec, X::AbsMat)
    return -((size(X, 1) * log(2π) + 2 * sum(ls)) .+ vec(sum(abs2.((X .- μ) ./ exp.(ls)), dims=1))) ./ 2
end










end # module

using Flux
using Distributions

m = Chain(Dense(2,128),Dense(128,128),Dense(128,1))
x=rand(2,1)
ps=Flux.params(m)
gs = gradient(() -> sum(m(x)), Flux.params(m))
G = M.gradvec(gs, ps)

pi = M.DiagonalGaussianPolicy(Chain(Dense(2,3),Dense(3,3,tanh),Dense(3, 3)), Float32[0.1,0.2,0.3])
ps=Flux.params(pi)
O = rand(2,100)
A = rand(3,100)
o = O[:, 1]    #
a = A[:, 1]

v=pdf(pi, o, a)
m=pdf(pi, O, A)
