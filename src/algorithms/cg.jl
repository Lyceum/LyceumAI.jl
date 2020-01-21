struct CG{T <: AbstractFloat}
    u::Vector{T}
    r::Vector{T}
    c::Vector{T}
    function CG{T}(dim_b::Integer, dim_x::Integer) where {T <: AbstractFloat}
        new{T}(zeros(T, dim_b), zeros(T, dim_b), zeros(T, dim_x))
    end
end
CG{T}(dims_A::Dims{2}) where {T} = CG{T}(dims_A...)

function (cg::CG{T})(x, A, b; tol::Real = sqrt(eps(real(eltype(b)))), maxiter::Int = size(A, 2), initiallyzero::Bool = false) where {T}
    if !(T == eltype(x) == eltype(A) == eltype(b))
        @warn "eltype mismatch: expected x, A, and b to have eltype $T. CG may run slower as a result" maxlog=1
        error()
    end

    d1, d2 = size(A)
    u = cg.u
    r = cg.r
    c = cg.c

    if !(length(r) == d1 && length(u) == length(c) == d2)
        resize!(u, d2)
        resize!(r, d1)
        resize!(c, d2)
    end

    fill!(u, zero(T))
    copyto!(r, b)
    # do not need to zero out c as e overwrite it below

    if initiallyzero
        # ignore x and treat x0 = 0 --> r0 = b
        residual = norm(b)
        reltol = residual * tol
    else
        # x0 = x --> r0 = b - A*x0
        mul!(c, A, x) # A*x0
        r .-= c # b - A*x0
        residual = norm(r)
        reltol = norm(b) * tol # TODO norm(b) or norm(r)?
    end

    prev_residual = one(residual)
    for _ = 1:maxiter
        residual <= reltol && break

        β = residual^2 / prev_residual^2
        u .= r .+ β .* u

        mul!(c, A, u)
        α = residual^2 / dot(u, c)

        # Improve solution and residual
        x .+= α .* u
        r .-= α .* c

        prev_residual = residual
        residual = norm(r)
    end
    x
end

_cg_tol(T::Type) = sqrt(eps(real(T)))