"""
    FVP{T <: Number, A <: AbstractMatrix{T}}

# Constructors
    FVP{T}(glls[, normalize = false])

# Functions
    Base.:*(op::FVP, x)
    LinearAlgebra.mul!(y, op::FVP)

Fisher Vector Product operator that handles internal caching. If `normalize` is true,
vector products will
"""
struct FVP{T <: Number, A <: AbstractMatrix}
    glls::A
    cache::Vector{T}
    normalize::Bool
    function FVP{T,A}(glls, normalize::Bool = false) where {T <: Number, A <: AbstractMatrix}
        glls = convert(AbstractMatrix{T}, glls)
        cache = Vector{T}(undef, size(glls, 2))
        new{T, typeof(glls)}(glls, cache, normalize)
    end
end

FVP{T}(glls, normalize = false) where {T} = FVP{T, typeof(glls)}(glls, normalize)
FVP(glls, normalize = false) = FVP{eltype(glls), typeof(glls)}(glls, normalize)

Base.size(op::FVP) = (size(op.glls, 1), size(op.glls, 1))
Base.size(op::FVP, d::Integer) = d > 3 ? 1 : length(op.cache)
Base.eltype(::Type{<:FVP{T}}) where {T} = T

"""
    *(op::FVP, x)

Return `op.glls * (tranpose(op.glls) * x)`
"""
function Base.:*(op::FVP, x::AbstractVector)
    T = Base.promote_op(*, eltype(op.glls), eltype(x))
    y = Vector{T}(undef, length(x))
    mul!(y, op, x)
end

"""
    mul!(y, op::FVP, x[, normalize = false])

Update `y` as `op.glls * (transpose(op.glls) * x)`
"""
function LinearAlgebra.mul!(y::AbstractVector, op::FVP, x::AbstractVector)
    alpha = op.normalize ? one(eltype(op)) / size(op.glls, 2) : true
    mul!(op.cache, transpose(op.glls), x)
    mul!(y, op.glls, op.cache, alpha, false)
end