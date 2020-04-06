module LyceumAI

# stdlib
using Random
using Random: default_rng
using Statistics, LinearAlgebra

# 3rd party
import LearnBase
using DocStringExtensions
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
using LyceumCore
using Zygote: Params, Grads
using Base: promote_eltype, @propagate_inbounds, require_one_based_indexing
using MLDataPattern: eachbatch, nobs
using MacroTools: @forward
using SpecialArrays
using LyceumBase: @mustimplement

zerofn(args...) = 0
noop(args...) = nothing


export
       # Algorithms
      MPPI,
      NaturalPolicyGradient,

       # Models
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
include("vectorproducts.jl")

include("flux.jl")


include("models/policy.jl")
export DiagGaussianPolicy, grad_loglikelihood!, loglikelihood

abstract type AbstractTrainer end
# (o::AbstractTrainer{M})(m::M) where M
#@mustimplement fit!(m::AbstractModel, data); export fit!
# --

include("algorithms/cg.jl")
include("algorithms/MPPI.jl")
include("algorithms/naturalpolicygradient.jl")


include("controller.jl")

end # module
