nparams(m) = (ps = params(m); length(ps) > 0 ? sum(length, ps) : 0)

function gradientandvalue(f, ps::Zygote.Params)
    y, back = Zygote.pullback(f, ps)
    losscheck(y)
    gs = back(Zygote.sensitivity(y))
    y, gs
end


####
#### FluxTrainerIterator
####

Base.@kwdef struct FluxTrainer{L, O, CB}
    lossfn::L = nothing
    optimiser::O = ADAM()
    szbatch::Union{Nothing,Int} = nothing
    batchdim::Union{Int,LearnBase.ObsDimension} = LearnBase.ObsDim.Last()
    stopcb::CB = infinitecb
end

function (p::FluxTrainer)(model, datas::Vararg{<:AbstractArray}; kwargs...)
    lossfn = haskey(kwargs, :lossfn) ? kwargs[:lossfn] : p.lossfn
    optimiser = haskey(kwargs, :optimiser) ? kwargs[:optimiser] : p.optimiser
    szbatch = haskey(kwargs, :szbatch) ? kwargs[:szbatch] : p.szbatch
    batchdim = haskey(kwargs, :batchdim) ? kwargs[:batchdim] : p.batchdim
    stopcb = haskey(kwargs, :stopcb) ? kwargs[:stopcb] : p.stopcb

    isnothing(lossfn) && error("`lossfn` not specified and no default provided")
    !(isnothing(szbatch) || szbatch isa Integer) && error("`szbatch` must be `nothing` or <:Integer")

    datas = isnothing(szbatch) ? (datas,) : eachbatch(datas, szbatch, batchdim)
    FluxTrainerIterator(model, datas, optimiser, lossfn, batchdim, stopcb)
end

struct FluxTrainerIterator{M,D,O,L,DIM,CB}
    model::M
    datas::D
    optimiser::O
    lossfn::L
    batchdim::DIM
    stopcb::CB
end

struct FluxTrainerState
    nepochs::Int
    nbatches::Int
    nsamples::Int
    avgloss::Float64
end
FluxTrainerState() = FluxTrainerState(0, 0, 0, 0)

function Base.iterate(ft::FluxTrainerIterator, state = FluxTrainerState())
    ft.stopcb(state) && return nothing

    model = ft.model
    nbatches = 0
    nsamples = 0
    totalloss = 0.0
    ps = params(ft.model)

    loss = let l=ft.lossfn, m=model
        (d...) -> l(m, d...)
    end

    for batch in ft.datas
        batchloss, gs = let batch=batch, ps=ps
            gradientandvalue(() -> loss(batch...), ps)
        end
        Flux.Optimise.update!(ft.optimiser, ps, gs)

        nbatches += 1
        nsamples += nobs(first(batch))
        totalloss += Float64(batchloss)
    end

    newstate = FluxTrainerState(
        state.nepochs + 1,
        state.nbatches + nbatches,
        state.nsamples + nsamples,
        totalloss / nsamples,
    )

    newstate, newstate
end



####
#### Utils
####
function multilayer_perceptron(
    d1,
    d2,
    d...;
    σ = identity,
    σ_final = identity,
    initW = Flux.glorot_uniform,
    initb = zeros,
    initW_final = Flux.glorot_uniform,
    initb_final = zeros,
    dtype = Float32,
)

    layers = []
    ds = (d1, d2, d...)
    for i = 2:(length(ds)-1)
        in, out = ds[i-1], ds[i]
        push!(layers, Flux.Dense(dtype.(initW(out, in)), dtype.(initb(out)), σ))
    end

    in, out = ds[end-1], ds[end]
    push!(
        layers,
        Flux.Dense(dtype.(initW_final(out, in)), dtype.(initb_final(out)), σ_final),
    )
    @assert length(layers) == length(ds) - 1

    Flux.Chain(layers...)
end

function flatupdate!(m, gs::AbstractVector)
    np = nparams(m)
    np == length(gs) || throw(ArgumentError("length(gs) != number of parameters"))

    from = firstindex(gs)
    @uviews gs for p in params(m)
        p = vec(Zygote.unwrap(p))
        to = from + length(p) - 1
        p .+= uview(gs, from:to)
        from += length(p)
    end
    m
end

function copygrad!(flatgrad::AbstractVector, ps::Zygote.Params, gs::Grads)
    from = firstindex(flatgrad)
    for p in ps
        g = gs[p]
        g === nothing && continue
        copyto!(flatgrad, from, g, firstindex(g), length(g))
        from += length(g)
    end
    flatgrad
end

# TODO support more than Matrix
function orthonormal(d1::Integer, d2::Integer)
    d1 > 0 && d2 > 0 || error("d1 and d2 must be > 0")

    shape = (d1, d2)
    A = randn(shape)
    U, _, Vt = svd(A)
    A = size(U) == shape ? U : Array(Vt')
    @assert size(A) == shape
    @assert !any(isnan, A)
    A
end


promote_modeltype(m, ms...) = promote_type(promote_modeltype(m), promote_modeltype(ms...))
function promote_modeltype(m)
    ls = filter(x -> x isa Union{Number,AbstractArray}, leaves(m))
    ls = map(Zygote.unwrap, ls) # iff any of ls is a TrackedArray/TrackedReal
    if length(ls) == 0
        throw(ArgumentError("m has no leaves <: Union{Number, AbstractArray}"))
    elseif length(ls) == 1
        return _paramtype(first(ls))
    else
        T = _paramtype(first(ls))
        for i = 2:length(ls)
            T = promote_type(T, _paramtype(ls[i]))
        end
        return T
    end
end

function _paramtype(x)
    if x isa Number
        return typeof(x)
    elseif x isa AbstractArray
        return eltype(x)
    else
        throw(ArgumentError("Expected x <: Union{Number, AbstractArray}"))
    end
end


function losscheck(x)
    x isa Real || error("Function output is not scalar")
    isinf(x) && error("Loss is infinite")
    isnan(x) && error("Loss is NaN")
end


function _leaves1(f, x)
    func, re = Flux.functor(x)
    map(f, func)
end

function leaves(x; cache = IdDict())
    haskey(cache, x) && return cache[x]
    cache[x] = Flux.isleaf(x) ? x : _leaves1(x -> leaves(x, cache = cache), x)
    collect(values(cache))
end
