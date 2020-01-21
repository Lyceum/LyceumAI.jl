@testset "`NaturalPolicyGradient` (PointMass)" begin
    seed_threadrngs!(1)
    etype = LyceumMuJoCo.PointMass

    e = etype()
    dobs, dact = length(obsspace(e)), length(actionspace(e))

    DT = Float32
    Hmax, K = 300, 32
    N = Hmax * K

    policy = DiagGaussianPolicy(
        multilayer_perceptron(dobs, 16, 16, dact, σ=tanh),
        zeros(dact)
    )
    policy = Flux.paramtype(DT, policy)

    value = multilayer_perceptron(dobs, 16, 16, 1, σ=Flux.relu)
    valueloss(bl, X, Y) = mse(vec(bl(X)), vec(Y))

    valuetrainer = FluxTrainer(
        optimiser = ADAM(1e-3),
        szbatch = 32,
        lossfn = valueloss,
        stopcb = s->s.nepochs > 4
    )
    value = Flux.paramtype(DT, value)

    npg = NaturalPolicyGradient(
        n -> tconstruct(etype, n),
        policy,
        value,
        gamma = 0.9,
        gaelambda = 0.95,
        valuetrainer,
        Hmax=Hmax,
        norm_step_size=0.05,
        N=N,
    )

    x = Float64[]
    for (i, state) in enumerate(npg)
        i > 50 && break
        push!(x, state.meanterminal_eval)
    end
    @test mean(x[(end-10):end]) < 0.15
end