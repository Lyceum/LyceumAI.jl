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

const AbsVecOrMat{T} = Union{AbstractVector{T},AbstractMatrix{T}} # TODO

#include("util/misc.jl")
#include("vectorproducts.jl")


#FluxTrainer, FluxTrainerIterator, orthonormal, multilayer_perceptron,
#include("flux.jl")


include("FluxTools.jl")
using .FluxTools


include("policy.jl")
export DiagonalGaussianPolicy



#include("algorithms/cg.jl")

#export MPPI
#include("algorithms/MPPI.jl")

#export NaturalPolicyGradient
#include("algorithms/naturalpolicygradient.jl")


export ControllerIterator
#include("controller.jl")

end # module
