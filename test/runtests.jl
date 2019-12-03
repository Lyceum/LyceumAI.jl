using Test, Pkg, LyceumAI, LyceumMuJoCo, LinearAlgebra, LyceumBase, Statistics
using LyceumBase.Tools

seed_threadrngs!(1)

@testset "LyceumAI.jl" begin

    @testset "util" begin
        include("util.jl")
    end

    @testset "MPPI (PointMass)" begin
        etype = LyceumMuJoCo.PointMass
        env = etype()
        T = 1000
        K = 32
        H = 25

        mppi = MPPI(
            sharedmemory_envctor = (i)->sharedmemory_envs(etype, i),
            covar0 = Diagonal(0.001^2*I, size(actionspace(env), 1)),
            lambda = 0.005,
            K =  K,
            H = H,
            gamma = 0.99
        )

        s = Array(undef, statespace(env))
        a = Array(undef, actionspace(env))
        o = Array(undef, observationspace(env))
        for t = 1:T
            getstate!(s, env)
            getobs!(o, env)
            getaction!(a, s, o, mppi)
            step!(env, a)
        end
        @test abs(geteval(env)) < 0.001
    end

end
