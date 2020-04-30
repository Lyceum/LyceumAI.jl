module TestFluxTools

include("preamble.jl")

using LyceumAI.FluxTools

const INLEN = 2
const OUTLEN = 3

function randmodel()
    W1 = rand(4, INLEN)
    b1 = rand(4)
    W2 = rand(OUTLEN, 4)
    b2 = rand(OUTLEN)
    m = Chain(Dense(W1, b1), Dense(W2, b2))
    @assert m[1].W === W1 && m[1].b === b1 && m[2].W === W2 && m[2].b === b2
    t = (W1=W1, b1=b1, W2=W2, b2=b2)
    return m, mapreduce(vec, vcat, t)
end

function randgrads(ps::Params)
    ids = Base.IdDict{Any,Any}()
    G = Vector{parameltype(ps)}()
    for (i, p) in enumerate(ps)
        ids[p] = rand(size(p)...)
        append!(G, ids[p])
    end
    Grads(ids), G
end

@testset "Params" begin
    let (m, P0) = randmodel()
        ps = params(m)
        @test paramvec(ps) isa Vector{parameltype(ps)}
        @test length(paramvec(ps)) == paramlength(ps)
        @test P0 == paramvec(ps)
    end
    let (m, P0) = randmodel()
        P = paramvec(params(m))
        P .= 0
        @test paramvec(params(m)) == P0
    end
    let (m, P0) = randmodel()
        P = paramvecview(params(m))
        P .= 0
        @test all(isequal(0), paramvec(params(m)))
    end
    let (ma, P0a) = randmodel(), (mb, P0b) = randmodel()
        psa = params(ma)
        psb = params(mb)

        copyparams!(psa, psb)
        @test paramvec(psa) == paramvec(psb)
        rand!(P0b)
        copyparams!(psa, P0b)
        @test paramvec(psa) == P0b
        rand!(P0b)
        copyparams!(P0b, psa)
        @test P0b == paramvec(psa)
    end
    let (m, P) = randmodel()
        ps = params(m)
        for p in ps
            p .= 0
        end

        gs, _ = randgrads(ps)
        updateparams!(ps, gs)
        @test all(p -> p == -gs[p], ps)

        for p in ps
            p .= 0
        end
        updateparams!(ps, P)
        @test paramvec(ps) == -P
    end
end

@testset "Grads" begin
    let (m, P) = randmodel()
        ps = params(m)
        gs, G0 = randgrads(ps)

        @test gradvec(gs, ps) isa Vector{parameltype(ps)}
        @test length(gradvec(gs, ps)) == gradlength(gs, ps)
        @test G0 == gradvec(gs, ps)
    end
    let (m, P0) = randmodel()
        ps = params(m)
        gs, G0 = randgrads(ps)
        G = gradvec(gs, ps)
        G .= 0
        @test gradvec(gs, ps) == G0
    end
    let (m, P0) = randmodel()
        ps = params(m)
        gs, G0 = randgrads(ps)
        G = gradvecview(gs, ps)
        G .= 0
        @test all(isequal(0), gradvec(gs, ps))
    end
    let (ma, P0a) = randmodel(), (mb, P0b) = randmodel()
        psa = params(ma)
        psb = params(mb)
        gsa, G0a = randgrads(psa)
        gsb, G0b = randgrads(psb)

        rand!(G0b)
        copygrads!(gsa, G0b, psa)
        @test gradvec(gsa, psa) == G0b
        rand!(G0b)
        copygrads!(G0b, gsa, psa)
        @test G0b == gradvec(gsa, psa)
    end
    let (m, P0) = randmodel()
        x = rand(INLEN)
        ya, gsa = value_and_gradient(() -> sum(m(x)), params(m))
        @test ya == sum(m(x))
        gsb = gradient(() -> sum(m(x)), params(m))
        @test all(params(m)) do p
            gsa[p] == gsb[p]
        end
    end
end

end # module