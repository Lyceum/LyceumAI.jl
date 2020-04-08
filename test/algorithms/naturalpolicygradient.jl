@testset "NaturalPolicyGradient (PointMass)" begin
    tseed!(1)
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

    envsampler = EnvironmentSampler(n -> tconstruct(etype, n))

    # Test that the terminal reward for the mean policy is > 0.95 for at least 5
    # iterations in a row, in at most 250 iterations.
    npasses = 0
    for (i, state) in enumerate(npg)
        (npasses > 5 || i > 250) && break
        batch = sample(envsampler, N, reset! = randreset!, Hmax=Hmax) do a, s, o
            a .= policy(o)
        end
        npasses = mean(τ -> τ.R[end], batch) > 0.95 ? npasses + 1 : 0
    end
    @test npasses > 5
end