module LyceumAI

using Random
using Random: default_rng
using Statistics, LinearAlgebra

# 3rd party
import LearnBase
using MLDataPattern: MLDataPattern, shuffleobs, eachbatch, nobs

using Base: @propagate_inbounds, require_one_based_indexing
using DocStringExtensions

using Distributions
using DistributionsAD

using EllipsisNotation # TODO
using FastClosures

using Flux
using Flux: @functor, params
using Zygote

using IterativeSolvers
using LyceumBase
using MacroTools
using Parameters

using Random
using Random: default_rng

using Shapes
using SpecialArrays
using Statistics: Statistics, mean, mean!, var, cov, cor
using StructArrays
using UnsafeArrays
using Zygote: Params, Grads

const AbsVec{T} = AbstractVector{T}
const AbsMat{T} = AbstractMatrix{T}
const AbsArr{T} = AbstractArray{T}
const AbsVecOrMat{T} = Union{AbstractVector{T},AbstractMatrix{T}} # TODO

include("misc.jl")
include("vectorproducts.jl")

export FluxTrainer, FluxTrainerIterator, orthonormal, multilayer_perceptron
include("flux.jl")

#@include("FluxTools.jl")
#@using .FluxTools

#include("abstractpolicy.jl") # TODO move to LyceumBase
#include("policy.jl")
include("oldpolicy.jl")
export DiagGaussianPolicy


include("algorithms/cg.jl")

export MPPI
include("algorithms/MPPI.jl")

export NaturalPolicyGradient
include("algorithms/naturalpolicygradient.jl")


#export ControllerIterator
#include("controller.jl")

end # module
