module TestNaturalPolicyGradient

include("../preamble.jl")

@testset "PointMass" begin
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
        mean_trajectory_buffer = rollout(envsampler, N, reset! = randreset!, Hmax=Hmax) do a, o
            a .= policy(o)
        end

        mean_trajectories = StructArray(mean_trajectory_buffer)

        npasses = mean(τ -> τ.R[end], mean_trajectories) > 0.95 ? npasses + 1 : 0
    end

    @test npasses > 5
end

@testset "HopperV2" begin
    tseed!(2)

    e = LyceumMuJoCo.HopperV2()
    dobs, dact = length(observationspace(e)), length(actionspace(e))

    DT = Float32
    Hmax = 1024
    K = 10
    N = Hmax * K

    policy = DiagGaussianPolicy(
        multilayer_perceptron(dobs, 64, 64, dact, σ=Flux.tanh),
        ones(dact) .*= -0.5
    )
    policy = Flux.paramtype(DT, policy)
    policy.meanNN[end].W .*= 1e-2
    policy.meanNN[end].b .*= 1e-2

    value = multilayer_perceptron(dobs+3, 128, 128, 1, σ=Flux.relu)

    valuetrainer = FluxTrainer(
        optimiser = ADAM(1e-3),
        szbatch = 64,
        lossfn = (bl, X, Y) -> mse(vec(bl(X)), vec(Y)),
        stopcb = s->s.nepochs > 2
    )
    value = Flux.paramtype(DT, value)

    timefeatureizer = LyceumAI.TimeFeatures{DT}(
        [1, 2, 3],
        [1, 1, 1],
        1 / 1000
    )

    npg = NaturalPolicyGradient{DT}(
        n -> tconstruct(LyceumMuJoCo.HopperV2, n),
        policy,
        value,
        gamma = 0.995,
        gaelambda = 0.97,
        valuetrainer,
        Hmax=Hmax,
        max_cg_iter = 12,
        norm_step_size=0.1,
        N=N,
        value_feature_op = timefeatureizer
    )

    mean_envsampler = EnvironmentSampler(n -> tconstruct(LyceumMuJoCo.HopperV2, n))
    meanR = Float64[]
    pass = false
    for (i, state) in enumerate(npg)

        mean_trajectory_buffer = rollout(mean_envsampler, N, reset! = randreset!, Hmax=Hmax) do a, o
            a .= policy(o)
        end
        mean_trajectories = StructArray(mean_trajectory_buffer)

        @info mean(τ -> sum(τ.R), mean_trajectories) > 2750
        pass = mean(τ -> sum(τ.R), mean_trajectories) > 2750
        (pass || i > 150) && break
    end

    @test pass
end


end
