@testset "NaturalPolicyGradient (PointMass)" begin
    tseed!(2)
    etype = LyceumMuJoCo.PointMass

    e = etype()
    dobs, dact = length(observationspace(e)), length(actionspace(e))

    DT = Float32
    Hmax, K = 256, 16
    N = Hmax * K

    policy = DiagGaussianPolicy(
        multilayer_perceptron(dobs, 16, 16, dact, σ=tanh),
        zeros(dact)
    )
    policy = Flux.paramtype(DT, policy)

    value = multilayer_perceptron(dobs, 32, 32, 1, σ=Flux.relu)
    valueloss(bl, X, Y) = mse(vec(bl(X)), vec(Y))

    valuetrainer = FluxTrainer(
        optimiser = RADAM(1e-3),
        szbatch = 64,
        lossfn = valueloss,
        stopcb = s->s.nepochs > 1
    )
    value = Flux.paramtype(DT, value)


    npg = NaturalPolicyGradient(
        n -> tconstruct(etype, n),
        policy,
        value,
        gamma = 0.97,
        gaelambda = 0.95,
        valuetrainer,
        Hmax=Hmax,
        norm_step_size=0.01,
        N=N,
    )

    meanR = Float64[]
    for (i, state) in enumerate(npg)
        i > 50 && break
        push!(meanR, mean(τ -> τ.R[end], state.batch.mean))
    end
    @test mean(meanR[(end-10):end]) > 0.85
end