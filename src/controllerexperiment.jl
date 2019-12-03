"""Convenience iterable for state-dependent controllers"""
struct ControllerExperiment{C,E,B}
    controller::C
    env::E
    T::Int
    buffer::B
    plotiter::Union{Int,Nothing}

    function ControllerExperiment(
        controller,
        env::AbstractEnv;
        T = 1000,
        plotiter = nothing,
    )
        buffer = TrajectoryBuffer(env, T)
        new{typeof(controller),typeof(env),typeof(buffer)}(
            controller,
            env,
            T,
            buffer,
            plotiter,
        )
    end
end

function Base.iterate(m::ControllerExperiment, t)
    if t > m.T
        return
    end

    rolloutstep!(m.controller, m.buffer.trajectory, m.env)
    if !isnothing(m.plotiter) && mod(t, m.plotiter) == 0
        rew = Tools.Line(m.buffer.trajectory.rewards, "Reward")
        eval = Tools.Line(m.buffer.trajectory.evaluations, "Eval")
        plt = Tools.expplot(
            rew,
            eval,
            title = "ControllerExperiment Iteration=$t/$(m.T)",
            width = 40,
        )
        display(plt)
    end

    (t, m), t + 1
end

Base.iterate(m::ControllerExperiment) = (reset!(m.env); reset!(m.controller); iterate(m, 1))
Base.length(m::ControllerExperiment) = m.T

function rolloutstep!(controller, traj::ElasticBuffer, env::AbstractEnv)
    grow!(traj)
    t = lastindex(traj)
    @uviews traj begin
        st, ot, at = view(traj.states, :, t),
            view(traj.observations, :, t),
            view(traj.actions, :, t)
        getstate!(st, env)
        getobs!(ot, env)
        getaction!(at, st, ot, controller)
        step!(env, at)
        traj.rewards[t] = getreward(env)
        traj.evaluations[t] = geteval(env)
    end
    traj
end

function terminate!(term::ElasticBuffer, env::AbstractEnv, trajlength::Integer, done::Bool)
    grow!(term)
    i = lastindex(term)
    @uviews term begin
        thistraj = view(term, i)
        getstate!(thistraj.states, env)
        getobs!(thistraj.observations, env)
        term.dones[i] = done
        term.lengths[i] = trajlength
    end
    term
end
