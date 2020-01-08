@testset "NPG (PointMass)" begin
    etype = LyceumMuJoCo.PointMass

    e = etype()
    dobs, dact = length(obsspace(e)), length(actionspace(e))

    DT = Float32
    Hmax, K = 300, 16
    N = Hmax * K

    policy = DiagGaussianPolicy(
        multilayer_perceptron(dobs, 32, 32, dact, σ=tanh),
        zeros(dact)
    )
    policy = Flux.paramtype(DT, policy)

    value = multilayer_perceptron(dobs, 32, 32, 1, σ=Flux.relu)
    valueloss(bl, X, Y) = mse(vec(bl(X)), vec(Y))

    valuetrainer = FluxTrainer(
        optimiser = ADAM(1e-2),
        szbatch = 32,
        lossfn = valueloss,
        stopcb = s->s.nepochs > 4
    )
    value = Flux.paramtype(DT, value)

    npg = NaturalPolicyGradient(
        n -> tconstruct(etype, n),
        policy,
        value,
        gamma = 0.95,
        gaelambda = 0.99,
        valuetrainer,
        Hmax=Hmax,
        norm_step_size=0.05,
        N=N,
    )

    meanterminal_eval = nothing
    for (i, state) in enumerate(npg)
        i > 30 && break
        meanterminal_eval = state.meanterminal_eval
        @info meanterminal_eval
    end

    @test meanterminal_eval < 0.1
end