using Test
using Pkg
using Statistics

using Flux

using LyceumBase
using LyceumMuJoCo
using LinearAlgebra
using UniversalLogger
using LyceumAI


function testrollout(getaction!, env::AbstractEnvironment, T)
    s = Array(undef, statespace(env))
    a = Array(undef, actionspace(env))
    o = Array(undef, observationspace(env))
    for t = 1:T
        getstate!(s, env)
        getobservation!(o, env)
        getaction!(a, s, o)
        setaction!(env, a)
        step!(env)
    end
    env
end


@testset "LyceumAI.jl" begin
    tseed!(1)

    @testset "algorithms" begin
        include("algorithms/MPPI.jl")
        include("algorithms/naturalpolicygradient.jl")
    end
    @testset "util" begin
        include("util/misc.jl")
    end
    @testset "vectorproducts" begin
        include("vectorproducts.jl")
    end

end
