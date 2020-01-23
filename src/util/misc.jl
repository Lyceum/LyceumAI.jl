export discount,
       GAEadvantages!,
       rollout!,
       threadedforeach,
       mse,
       nstep_returns!

const AbsVecOfVec = AbstractVector{<:AbstractVector}


####
#### Discounted N-step returns
####

@inline discount(reward::Real, gamma::Real, t::Integer) = gamma^(t - 1) * reward


nstep_returns!(returns::AbsVec, rewards::AbsVec, gamma) =
    bootstrapped_nstep_returns!(returns, rewards, 0, gamma)

@propagate_inbounds function nstep_returns!(
    returns::AbsVecOfVec,
    rewards::AbsVecOfVec,
    gamma::Real,
)
    @boundscheck begin
        if size(returns) != size(rewards)
            throw(ArgumentError("returns and rewards must have same size"))
        end
    end

    for (rets, rews) in zip(returns, rewards)
        @inbounds nstep_returns!(rets, rews, gamma)
    end
    returns
end

@propagate_inbounds function nstep_returns!(returns::AbsMat, rewards::AbsMat, gamma::Real)
    @boundscheck begin
        if axes(returns) != axes(rewards)
            throw(ArgumentError("returns and rewards must have same axes"))
        end
    end

    @uviews returns rewards for k in axes(returns, 2)
        @inbounds nstep_returns!(view(returns, :, k), view(rewards, :, k), gamma)
    end
    returns
end


@propagate_inbounds function bootstrapped_nstep_returns!(
    returns::AbsVec{<:Real},
    rewards::AbsVec{<:Real},
    terminal_value::Real,
    gamma::Real,
)
    @boundscheck begin
        if axes(returns) != axes(rewards)
            throw(ArgumentError("returns and rewards must have same axes"))
        end
        require_one_based_indexing(returns, rewards)
    end

    DT = eltype(returns)
    gamma, ret_t = DT(gamma), DT(terminal_value)
    @inbounds for t in reverse(eachindex(returns))
        returns[t] = ret_t = rewards[t] + gamma * ret_t
    end
    returns
end

@propagate_inbounds function bootstrapped_nstep_returns!(
    returns::AbsVecOfVec,
    rewards::AbsVecOfVec,
    terminal_values::AbsVec,
    gamma::Real,
)
    @boundscheck begin
        if !(length(returns) == length(rewards) == length(terminal_values))
            throw(ArgumentError("returns, rewards, and terminal_values must have same length"))
        end
    end

    for (rets, rews, term) in zip(returns, rewards, terminal_values)
        @inbounds bootstrapped_nstep_returns!(rets, rews, term, gamma)
    end
    returns
end

@propagate_inbounds function bootstrapped_nstep_returns!(
    returns::AbsMat,
    rewards::AbsMat,
    terminal_values::AbsVec,
    gamma,
)
    @boundscheck begin
        if axes(returns) != axes(rewards) || axes(returns, 2) != axes(terminal_values, 1)
            throw(ArgumentError("returns and rewards must have same axes and share 2nd axis with terminal_values"))
        end
        require_one_based_indexing(returns, rewards, terminal_values)
    end

    @uviews returns rewards for k in axes(returns, 2)
        @inbounds bootstrapped_nstep_returns!(
            view(returns, :, k),
            view(rewards, :, k),
            terminal_values[k],
            gamma,
        )
    end
    returns
end


####
#### Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
####

@propagate_inbounds function GAEadvantages!(
    advantages::AbsVec{<:Real},
    values::AbsVec{<:Real},
    rewards::AbsVec{<:Real},
    terminal_value::Real,
    gamma::Real,
    lambda::Real,
)
    @boundscheck begin
        if !(axes(advantages) == axes(values) == axes(rewards))
            throw(BoundsError("advantages, values, and rewards must have same axes"))
        end
        require_one_based_indexing(advantages, values, rewards)
    end
    DT = eltype(advantages)
    gamlam = DT(gamma * lambda)
    @inbounds advantages[end] = rewards[end] + gamma * terminal_value - values[end]
    @inbounds advsum = adv_t = advantages[end] # t=T
    @inbounds for t in reverse(eachindex(advantages)[1:end-1]) # t=(T-1):-1:1
        td_error = rewards[t] + gamma * values[t+1] - values[t]
        advantages[t] = adv_t = td_error + gamlam * adv_t
        advsum += adv_t
    end
    advantages
end

@propagate_inbounds function GAEadvantages!(
    advantages::AbsVecOfVec,
    values::AbsVecOfVec,
    rewards::AbsVecOfVec,
    terminal_values::AbsVec,
    gamma,
    lambda,
)
    @boundscheck if !(size(advantages) == size(values) == size(rewards) ==
                      size(terminal_values))
        throw(BoundsError("advantages, values, rewards, and terminal_values must have same size"))
    end
    for (a, v, r, term) in zip(advantages, values, rewards, terminal_values)
        @inbounds GAEadvantages!(a, v, r, term, gamma, lambda)
    end
    advantages
end

@propagate_inbounds function GAEadvantages!(
    advantages::AbsMat,
    values::AbsMat,
    rewards::AbsMat,
    terminal_values::AbsVec,
    gamma,
    lambda,
)
    @boundscheck begin
        if !(axes(advantages) == axes(values) == axes(rewards) && axes(
            advantages,
            2,
        ) == axes(terminal_values, 1))
            throw(ArgumentError("advantages, values, and rewards must have matching axes and share 2nd axis with terminal_values"))
        end
    end

    @uviews advantages values rewards for k in axes(advantages, 2)
        @inbounds GAEadvantages!(
            view(advantages, :, k),
            view(values, :, k),
            view(rewards, :, k),
            terminal_values[k],
            gamma,
            lambda,
        )
    end
    advantages
end


####
#### Miscellaneous
####

mse(ŷ::AbstractMatrix, y::AbstractMatrix) = sum((ŷ - y) .^ 2) * 1 // length(y)
mse(ŷ::AbstractVector, y::AbstractVector) = sum((ŷ - y) .^ 2) * 1 // length(y)

whiten(x::AbstractArray) = whiten(copy(x))
function whiten!(x::AbstractArray)
    μ, σ = mean(x), std(x)
    x .= (x .- μ) ./ (σ + eps(eltype(x)))
end


struct TimeFeatures{T}
    orders::Vector{T}
    coeffs::Vector{T}
    timestep::T
    function TimeFeatures{T}(orders::AbsVec, coeffs::AbsVec, timestep::Real) where {T<:AbstractFloat}
        if length(orders) != length(coeffs)
            throw(ArgumentError("length(orders) must equal length(coeffs)"))
        end
        timestep > 0 || throw(ArgumentError("timestep must be > 0"))

        orders = convert(Vector{T}, orders)
        coeffs = convert(Vector{T}, coeffs)
        timestep = convert(T, timestep)

        new{T}(orders, coeffs, timestep)
    end
end

function TimeFeatures(orders::AbsVec, coeffs::AbsVec, timestep::Real)
    TimeFeatures{Float64}(orders, coeffs, timestep)
end

function (op::TimeFeatures)(B::AbstractMatrix, A::AbstractMatrix)
    Base.require_one_based_indexing(B, A)
    if size(B, 1) != size(A, 1) + length(op.coeffs)
        throw(DimensionMismatch("size(B, 1) != size(A, 1) + length(op.coeffs)"))
    end
    size(B, 2) != size(A, 2) && throw(DimensionMismatch("size(B, 2) != size(A, 2)"))

    T = eltype(B)
    dimobs, N = size(A)
    dimfeats = length(op.coeffs)
    x = (0:(N - 1)) .* op.timestep

    r1 = 1:dimobs
    r2 = (dimobs + 1):(dimobs + dimfeats)

    for j = axes(B, 2)
        a = uview(A, :, j)
        b1 = uview(B, r1, j)
        b2 = uview(B, r2, j)
        copyto!(b1, a)
        for k = 1:dimfeats
            i = k + dimobs
            B[i, j] = op.coeffs[k] * x[j]^op.orders[k]
        end
    end
    B
end

@inline function (op::TimeFeatures{T})(A::AbstractMatrix) where {T}
    V = promote_type(T, eltype(A))
    B = Matrix{V}(undef, size(A, 1) + length(op.coeffs), size(A, 2))
    op(B, A)
end

# TODO
# 1) inner_size w/ BatchedVector
# 2) return BatchedVector{<:NestedView}
function (op::TimeFeatures{T})(A::AbsVec{<:AbsMat}) where {T}
    for a in A
        if size(a, 1) != size(first(A), 1)
            error("The first axis of each element in A must be equal")
        end
    end

    N = sum(a -> size(a, 2), A)
    dimobs = size(first(A), 1)

    B = Matrix{T}(undef, dimobs + length(op.coeffs), N)
    from = firstindex(B, 2)
    for a in A
        l = size(a, 2)
        to = from + l - 1
        b = view(B, :, from:to)
        op(b, a)
        from += l
    end
    B
end

@inline function (op::TimeFeatures{T})(A::AbsMat{<:Real}, ts::AbsVec{<:Integer}) where {T}
    Base.require_one_based_indexing(A, ts)
    if size(A, 2) != length(ts)
        throw(DimensionMismatch("size(A, 2) != length(ts)"))
    end

    dimobs, N = size(A)
    dimfeats = length(op.coeffs)
    B = Matrix{T}(undef, dimobs + dimfeats, N)

    r1 = 1:dimobs
    r2 = (dimobs + 1):(dimobs + dimfeats)

    for j = axes(B, 2)
        a = uview(A, :, j)
        b1 = uview(B, r1, j)
        b2 = uview(B, r2, j)
        t = (ts[j] - 1) * op.timestep
        copyto!(b1, a)
        for k = 1:dimfeats
            i = k + dimobs
            B[i, j] = op.coeffs[k] * t^op.orders[k]
        end
    end
    B
end