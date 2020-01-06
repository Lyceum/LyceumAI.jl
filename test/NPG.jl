using
    LyceumBase, LyceumAI, LyceumMuJoCo,
    Flux, LinearAlgebra, Random, UnicodePlots, EllipsisNotation, Statistics, FastClosures, LyceumBase, LinearAlgebra, UniversalLogger, JLSO, StaticArrays

using LyceumBase.Tools
using LyceumBase.Tools: filter_nt

function runNPG(etype, plotiter=1)
    #Random.seed!(1) # TODO all threads
    seed_threadrngs!()

    BLAS.set_num_threads(Threads.nthreads())

    e = etype()
    dobs, dact = length(obsspace(e)), length(actionspace(e))

    DT = Float32
    Hmax, K = 500, 16
    N = Hmax * K

    policy = DiagGaussianPolicy(
        multilayer_perceptron(dobs, 64, 64, dact, σ=tanh),
        zeros(dact),
    )
    policy=Flux.paramtype(Float64, policy)

    value = multilayer_perceptron(dobs, 64, 64, 1, σ=Flux.elu)
    valueloss(bl, X, Y) = Flux.mse(vec(bl(X)), vec(Y))
    valuetrainer = FluxTrainer(
        optimiser = ADAM(1e-3),
        szbatch = div(Hmax*K, 200),
        stopcb = x -> x.nepochs > 3,
        lossfn = valueloss
    )
    value = Flux.paramtype(Float64, value)

    npg = NaturalPolicyGradient(
        (n) -> tconstruct(etype, n),
        policy,
        value,
        valuetrainer,
        Hmax=Hmax,
        norm_step_size=0.1,
        N=N,
    )

    envname = lowercase(string(nameof(etype)))
    savepath = "/tmp/$envname.jlso"
    exper = Experiment(savepath, overwrite=true)
    lg = ULogger()

    for (i, state) in enumerate(npg)
        if i > 30
            break
            exper[:policy] = npg.policy
            exper[:value] = npg.value
            exper[:etype] = etype
            exper[:meanstates] = state.meanbatch
            exper[:stocstates] = state.stocbatch
            break
        end

        push!(lg, :algstate, filter_nt(state, exclude=(:meanbatch, :stocbatch)))

        if mod(i, plotiter) == 0
            x = lg[:algstate]
            display(expplot(
                Line(x[:stocterminal_eval], "StocLastE"),
                Line(x[:meanterminal_eval], "MeanLastE"),
                title="NPG Iteration=$i", width=80, height=12
            ))

            display(expplot(
                Line(x[:stoctraj_reward], "StocR"),
                Line(x[:meantraj_reward], "MeanR"),
                title="NPG Iteration=$i", width=80, height=12
            ))
        end
    end
    npg, exper
end

include("new/swimmer.jl")
#runNPG(LyceumMuJoCo.SwimmerV2)
runNPG(NewSwimmer)

nothing





