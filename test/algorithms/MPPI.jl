@testset "MPPI (PointMass)" begin
    seed_threadrngs!(1)
    etype = LyceumMuJoCo.PointMass
    env = etype()
    T = 300
    K = 8
    H = 10

    mppi = MPPI(
        env_tconstructor = n -> tconstruct(etype, n),
        covar0 = Diagonal(0.1^2*I, size(actionspace(env), 1)),
        lambda = 0.01,
        K =  K,
        H = H,
        gamma = 0.99
    )
    env = testrollout(env, T) do a, s, o
        getaction!(a, s, o, mppi)
    end
    @test abs(geteval(env)) < 0.001
end

