"""Convenience iterable for state-dependent controllers"""
struct ControllerIterator{C,E,B}
    controller::C
    env::E
    T::Int
    trajectory::B
    plotiter::Int
    randstart::Bool

    function ControllerIterator(
        controller,
        env::AbstractEnvironment;
        T::Integer = 1000,
        plotiter::Integer = 1,
        randstart::Bool = false
    )
        trajectory = (
            states = Array(undef, statespace(env), T),
            observations = Array(undef, obsspace(env), T),
            actions = Array(undef, actionspace(env), T),
            rewards = Array(undef, rewardspace(env), T),
            evaluations = Array(undef, evalspace(env), T),
        )
        new{typeof(controller),typeof(env),typeof(trajectory)}(
            controller,
            env,
            T,
            trajectory,
            plotiter,
            randstart
        )
    end
end

function Base.iterate(it::ControllerIterator, t)
    t > it.T && return

    rolloutstep!(it.controller, it.trajectory, it.env, t)
    if mod(t, it.plotiter) == 0
        rew = @views Tools.Line(it.trajectory.rewards[1:t], "Reward")
        eval = @views Tools.Line(it.trajectory.evaluations[1:t], "Eval")
        plt = Tools.expplot(
            rew,
            eval,
            title = "ControllerIterator Iteration=$t/$(it.T)",
            width = 40,
        )
        display(plt)
    end

    (t, it), t + 1
end

function Base.iterate(it::ControllerIterator)
    it.randstart ? randreset!(it.env) : reset!(it.env)
    reset!(it.controller)
    iterate(it, 1)
end

Base.length(it::ControllerIterator) = it.T

function rolloutstep!(controller, traj, env, t)
    # TODO uviews
    st = view(traj.states, .., t)
    ot = view(traj.observations, .., t)
    at = view(traj.actions, .., t)

    getstate!(st, env)
    getobs!(ot, env)
    getaction!(at, st, ot, controller)
    setaction!(env, at)

    step!(env)
    r = getreward(st, at, ot, env)
    e = geteval(st, at, ot, env)
    done = isdone(st, at, ot, env)

    traj.rewards[t] = r
    traj.evaluations[t] = e

    traj
end