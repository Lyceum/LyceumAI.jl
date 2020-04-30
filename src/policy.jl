####
#### Model Interface
####

# (m)(xs...)
# Flux.params(m)
# Flux.functor(m)

# optional overrides
#paramvec(m) = paramvec(params(m))
#paramvecview(m) = paramvecview(params(m))
#parameltype(m) = parameltype(params(m))
#paramlength(m) = paramlength(params(m))

#copyparams!(ps::ParamTypes, m) = copyparams!(ps, params(m))
#copyparams!(m, ps::ParamTypes) = copyparams!(params(m), ps)

#updateparams!(m, gs::GradTypes) = updateparams!(params(m), gs)

#abstract type AbstractModel end


####
#### Policy
####




#struct StochasticPolicy{DistType,Params<:Tuple} <: AbstractStochasticPolicy
#    params::Params
#end

#function StochasticPolicy(::Type{DistType}, params::Params) where {DistType,Params<:Tuple}
#    StochasticPolicy{DistType,Params}(params)
#end
#StochasticPolicy(T::Type, params...) = StochasticPolicy(T, params)

#condition(π::StochasticPolicy, o::AbsVec) = _condition(π, o)
#Base.@generated function _condition(π::StochasticPolicy{DistType,Params}, o::AbsVec) where {DistType,Params}
#    N = length(Params.parameters)
#    argexprs = map(1:fieldcount(Params)) do i
#        fieldtype(Params, i) <: AbsVec ? :(π.params[$i]) : :(π.params[$i](o))
#    end
#    MacroTools.@q begin
#        @_inline_meta
#        $(Expr(:call, DistType, argexprs...))
#    end
#end


####
#### DiagonalGaussianPolicy
####

struct DiagonalGaussianPolicy{Mean,LogStd} <: AbstractStochasticPolicy
    mean::Mean
    logstd::LogStd
end

const FixedDiagonalGaussianPolicy = DiagonalGaussianPolicy{<:Any,<:AbsVec}

@inline _logstd(π::DiagonalGaussianPolicy, o::AbsVec) = exp.(π.logstd)
@inline _logstd(π::FixedDiagonalGaussianPolicy, o::AbsVec) = exp.(π.logstd(o))

Flux.@functor DiagonalGaussianPolicy

@inline condition(π::DiagonalGaussianPolicy, o::AbsVec) = MvNormal(π(o), _logstd(π, o))

(π::DiagonalGaussianPolicy)(o::AbsVecOrMat) = mean(π, o)

Statistics.mean(π::DiagonalGaussianPolicy, o::AbsVec) = π.mean(o)
Statistics.mean(π::DiagonalGaussianPolicy, O::AbsMat) = π.mean(O)


#function Distributions.logpdf(π::DiagonalGaussianPolicy, o::AbsVec, a::AbsVec)
#    _mvnormal_logpdf(mean(π, o), π.logstd(o), a)
#end
#function Distributions.logpdf(π::DiagonalGaussianPolicy, O::AbsMat, A::AbsMat)
#    _mvnormal_logpdf(mean(π, O), π.logstd(O), A)
#end

function Distributions.logpdf(π::FixedDiagonalGaussianPolicy, o::AbsVec, a::AbsVec)
    _mvnormal_logpdf(mean(π, o), π.logstd, a)
end
function Distributions.logpdf(π::FixedDiagonalGaussianPolicy, O::AbsMat, A::AbsMat)
    _mvnormal_logpdf(mean(π, O), π.logstd, A)
end

function _mvnormal_logpdf(μ::AbsVec, ls::AbsVec, x::AbsVec)
    return -(length(x) * log(2π) + 2 * sum(ls) + sum(abs2.((x .- μ) ./ exp.(ls)))) / 2
end

function _mvnormal_logpdf(μ::AbsMat, ls::AbsVec, X::AbsMat)
    return -((size(X, 1) * log(2π) + 2 * sum(ls)) .+ vec(sum(abs2.((X .- μ) ./ exp.(ls)), dims=1))) ./ 2
end