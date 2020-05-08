module LyceumAI

using Random
using Random: default_rng
using Statistics, LinearAlgebra

# 3rd party
#import LearnBase
#using MLDataPattern: eachbatch, nobs

using Base: @propagate_inbounds, require_one_based_indexing
using DocStringExtensions

using Distributions
using DistributionsAD

using EllipsisNotation
using FastClosures

using Flux
using Flux: @functor, params

using IterativeSolvers
using LyceumBase
using LyceumCore
using MacroTools
using Parameters

using Random
using Random: default_rng

using RecursiveArrayTools
#using Shapes
#using SpecialArrays
#using Statistics: Statistics, mean, mean!, var, cov, cor
#using StatsBase: StatsBase, cor, entropy
using UnsafeArrays
using Zygote: Params, Grads

#export
#       # Algorithms
#      MPPI,
#      NaturalPolicyGradient,

#       # Models
#      DiagGaussianPolicy,
#      grad_loglikelihood!,
#      loglikelihood,

#       # Flux tools
#      FluxTrainer,
#      FluxTrainerIterator,
#      orthonormal,
#      multilayer_perceptron,

#       # Miscellaneous
#      ControllerIterator

const AbsVecOrMat{T} = Union{AbstractVector{T},AbstractMatrix{T}}

#include("util/misc.jl")
#include("vectorproducts.jl")
#include("flux.jl")


include("FluxTools.jl")
using .FluxTools


include("policy.jl")
export DiagonalGaussianPolicy



#include("algorithms/cg.jl")
#include("algorithms/MPPI.jl")
#include("algorithms/naturalpolicygradient.jl")


#include("controller.jl")

end # module
