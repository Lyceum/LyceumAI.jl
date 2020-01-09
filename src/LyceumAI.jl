module LyceumAI

# stdlib
using Random, Statistics, LinearAlgebra

# 3rd party
import LearnBase
import Flux: params, params!
import Flux.Optimise: update!
using UnsafeArrays,
      StaticArrays,
      Distances,
      FastClosures,
      IterativeSolvers,
      Parameters,
      EllipsisNotation,
      Flux,
      Zygote,

      Shapes,
      LyceumBase

# Lyceum
import LyceumBase: getaction!
using LyceumBase.Tools
using LyceumBase.Tools: zerofn, noop

using Zygote: Params, Grads
using Base: promote_eltype, @propagate_inbounds, require_one_based_indexing
using MLDataPattern: eachbatch, nobs
using MacroTools: @forward
using LyceumBase: @mustimplement


export
       # Algorithms
      MPPI,
      NaturalPolicyGradient,

       # Models
      AbstractPolicy,
      DiagGaussianPolicy,
      grad_loglikelihood!,
      loglikelihood,

       # Flux tools
      FluxTrainer,
      FluxTrainerIterator,
      orthonormal,
      multilayer_perceptron,

       # Miscellaneous
      ControllerIterator


const AbsVec = AbstractVector
const AbsMat = AbstractMatrix

infinitecb(x...) = false



abstract type AbstractModel{T} end
Base.eltype(m::AbstractModel{T}) where {T} = T

@mustimplement params(m::AbstractModel)
@mustimplement params!(ps, m::AbstractModel)
@mustimplement params!(m::AbstractModel, ps)

@mustimplement update!(m::AbstractModel, gs)


include("util/misc.jl")

include("flux.jl")


include("models/policy.jl")
export AbstractPolicy, DiagGaussianPolicy, grad_loglikelihood!, loglikelihood

abstract type AbstractTrainer end
# (o::AbstractTrainer{M})(m::M) where M
#@mustimplement fit!(m::AbstractModel, data); export fit!
# --

include("algorithms/MPPI.jl")
include("algorithms/naturalpolicygradient.jl")


include("controller.jl")

end # module
