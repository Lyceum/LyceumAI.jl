"""Convenience iterable for state-dependent controllers"""
struct ControllerIterator{C,E,B}
    controller::C
    env::E
    T::Int
    trajectory::B
    plotiter::Int
    randstart::Bool
    alreadyran::Base.RefValue{Bool}

    function ControllerIterator(
        controller,
        env::AbstractEnvironment;
        T::Integer = 1000,
        plotiter::Integer = 1,
        randstart::Bool = false
    )
        trajectory = (
            states = Array(undef, statespace(env), T),
            observations = Array(undef, observationspace(env), T),
            actions = Array(undef, actionspace(env), T),
            rewards = Array(undef, rewardspace(env), T),
        )
        new{typeof(controller),typeof(env),typeof(trajectory)}(
            controller,
            env,
            T,
            trajectory,
            plotiter,
            randstart,
            Ref(false)
        )
    end
end

function Base.iterate(it::ControllerIterator, t::Int = 1)
    it.alreadyran[] && error("Cannot iterate on a ControllerIterator more than once")
    if t > it.T
        it.alreadyran[] = true
        return
    end

    rolloutstep!(it.controller, it.trajectory, it.env, t)
    if mod(t, it.plotiter) == 0
        display(termplot(it.trajectory.rewards[1:t], title = "Reward (t=$t/$(it.T)"))
    end

    (t, it), t + 1
end

Base.length(it::ControllerIterator) = it.T

function rolloutstep!(controller, traj, env, t)
    st = view(traj.states, .., t)
    ot = view(traj.observations, .., t)
    at = view(traj.actions, .., t)

    getstate!(st, env)
    getobservation!(ot, env)
    controller(at, st, ot)
    setaction!(env, at)

    step!(env)
    r = getreward(st, at, ot, env)
    done = isdone(st, ot, env)

    traj.rewards[t] = r

    traj
end
