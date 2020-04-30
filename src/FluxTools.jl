module FluxTools

using Base: IdSet, promote_eltype
using Flux: Flux, fmap, functor, params
using LyceumCore
using Zygote: Zygote, Params, Grads, Buffer, gradient, pullback, sensitivity
using RecursiveArrayTools: ArrayPartition


export destructure, restructure
export paramvec, paramvecview, parameltype, paramlength, copyparams!, updateparams!, allparams, pmap
export gradvec, gradvecview, gradeltype, gradlength, copygrads!, value_and_gradient


const NumVec{T<:Number} = AbstractVector{T}
const ParamTypes = Union{Params, NumVec}
const GradTypes = Union{Grads, NumVec}


####
#### Model
####

function destructure(m, ps = params(m))
    let ps = Params(ps)
        return reduce(vcat, vec.(copy(ps.order::Buffer))), P -> restructure(m, P, ps)
    end
end

function restructure(m, P::NumVec, ps = params(m))
    offset = firstindex(P)
    m = pmap(m, ps) do x
        x = reshape(P[offset:(offset + length(x) - 1)], size(x))
        offset += length(x)
        return x
    end
    offset == length(P) + 1 || throw(BoundsError())
    return m
end


####
#### Params
####

parameltype(ps::Params) = foldl(promote_eltype, ps)

paramlength(ps::Params) = sum(p -> length(p)::Int, ps)

paramvec(ps::Params) = reduce(vcat, vec.(ps))

paramvecview(ps::Params) = ArrayPartition(copy(ps.order)...)

copyparams!(ps1::Params, ps2::Params) = _copyto!(ps1, ps2)
copyparams!(ps::Params, P::NumVec) = _copyto!(ps, P)
copyparams!(P::NumVec, ps::Params) = _copyto!(P, ps)


function updateparams!(ps::Params, gs::Grads)
    for p in ps
        gs[p] === nothing && continue
        p .-= gs[p]
    end
    return ps
end

function updateparams!(ps::Params, G::NumVec)
    offset = firstindex(G)
    for p in ps
        g = reshape(view(G, offset:(offset + length(p) - 1)), size(p))
        p .-= g
        offset += length(p)
    end
    offset == length(G) + 1 || throw(BoundsError())
    return ps
end


allparams!(p::Params, x::AbsArr{<:Number}, seen = IdSet()) = push!(p, x)

function allparams!(p::Params, x, seen = IdSet())
    x in seen && return
    push!(seen, x)
    for child in first(functor(x))
        allparams!(p, child, seen)
    end
end

function allparams(m...)
    ps = Params()
    allparams!(ps, m)
    return ps
end


function pmap(f, m, ps = params(m))
    ps = Params(ps)
    fmap(x -> x in ps.params ? f(x) : x, m)
end


####
#### Grads
####

gradeltype(gs::Grads, ps::Params) = foldl(promote_eltype, extractgrads(gs, ps))

gradlength(gs::Grads, ps::Params) = sum(g -> length(g)::Int, extractgrads(gs, ps))

gradvec(gs::Grads, ps::Params) = reduce(vcat, vec.(extractgrads(gs, ps)))

gradvecview(gs::Grads, ps::Params) = ArrayPartition(extractgrads(gs, ps)...)

copygrads!(G::NumVec, gs::Grads, ps::Params) = _copyto!(G, extractgrads(gs, ps))
copygrads!(gs::Grads, G::NumVec, ps::Params) = _copyto!(extractgrads(gs, ps), G)

function value_and_gradient(f, args...)
    y, back = pullback(f, args...)
    return y, back(sensitivity(y))
end

extractgrads(gs::Grads, ps::Params) = [gs[p] for p in ps if gs[p] !== nothing]


####
#### Util
####

function _copyto!(xs1, xs2)
    for (x1, x2) in zip(xs1, xs2)
        size(x1) == size(x2) || error("Expected size $(size(p1)), got $(size(p2))")
        copyto!(x1, x2)
    end
    return xs1
end

function _copyto!(xs, V::NumVec)
    offset = firstindex(V)
    for x in xs
        copyto!(x, firstindex(x), V, offset, length(x))
        offset += length(x)
    end
    offset == length(V) + 1 || throw(BoundsError())
    return xs
end

function _copyto!(V::NumVec, xs)
    offset = firstindex(V)
    for x in xs
        copyto!(V, offset, x, firstindex(x), length(x))
        offset += length(x)
    end
    offset == length(V) + 1 || throw(BoundsError())
    return V
end


end # module