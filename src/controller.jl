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
        env::AbstractEnv;
        T = 1000,
        plotiter = 1,
        randstart = false
    )
        trajectory = (
            states = Array(undef, statespace(env), T),
            observations = Array(undef, observationspace(env), T),
            actions = Array(undef, actionspace(env), T),
            rewards = Array(undef, rewardspace(env), T),
            evaluations = Array(undef, evaluationspace(env), T),
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

function Base.iterate(m::ControllerIterator, t)
    t > m.T && return

    rolloutstep!(m.controller, m.trajectory, m.env, t)
    if mod(t, m.plotiter) == 0
        rew = @views Tools.Line(m.trajectory.rewards[1:t], "Reward")
        eval = @views Tools.Line(m.trajectory.evaluations[1:t], "Eval")
        plt = Tools.expplot(
            rew,
            eval,
            title = "ControllerIterator Iteration=$t/$(m.T)",
            width = 40,
        )
        display(plt)
    end

    (t, m), t + 1
end

Base.iterate(m::ControllerIterator) = (m.randstart ? randreset!(m.env) : reset!(m.env) ; reset!(m.controller); iterate(m, 1))
Base.length(m::ControllerIterator) = m.T

function rolloutstep!(controller, traj, env, t)
    st = view(traj.states, .., t)
    ot = view(traj.observations, .., t)
    at = view(traj.actions, .., t)

    getstate!(st, env)
    getobs!(ot, env)
    getaction!(at, st, ot, controller)

    r, e, done = step!(env, at)
    traj.rewards[t] = r
    traj.evaluations[t] = e

    traj
end
