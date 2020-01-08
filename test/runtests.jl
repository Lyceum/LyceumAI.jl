using Test
using Pkg
using Statistics

using Flux

using LyceumBase
using LyceumBase.Tools
using LyceumMuJoCo
using LinearAlgebra
using UniversalLogger
using LyceumAI


function testrollout(getaction!, env::AbstractEnvironment, T)
    s = Array(undef, statespace(env))
    a = Array(undef, actionspace(env))
    o = Array(undef, obsspace(env))
    for t = 1:T
        getstate!(s, env)
        getobs!(o, env)
        getaction!(a, s, o)
        setaction!(env, a)
        step!(env)
    end
    env
end


@testset "LyceumAI.jl" begin

    seed_threadrngs!(1)

    @testset "algorithms" begin
        include("algorithms/MPPI.jl")
        include("algorithms/NPG.jl")
    end

    @testset "util" begin
        include("util/misc.jl")
    end

end
