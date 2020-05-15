module TestMPPI

include("../preamble.jl")

@testset "MPPI (PointMass)" begin
    tseed!(1)
    etype = LyceumMuJoCo.PointMass
    env = etype()
    T = 300
    K = 8
    H = 10

    mppi = MPPI(
        env_tconstructor = n -> tconstruct(etype, n),
        covar = Diagonal(0.1^2*I, size(actionspace(env), 1)),
        lambda = 0.01,
        K =  K,
        H = H,
        gamma = 0.99
    )
    f = (a, s, o) -> getaction!(a, s, mppi)
    experiment = ControllerIterator(f, env; T=T, plotiter=T+1)
    for x in experiment
    end

    @test isapprox(getreward(env), 1, atol=0.001)
end

end
