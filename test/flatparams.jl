using Flux.Zygote
using Flux.Zygote: Buffer
using MacroTools: @forward
using Base: IdSet

function flatmodel(m)
    ps = Flux.params(m)
    psflat = mapreduce(vec, vcat, ps)
    offset = 0
    Flux.fmap(m) do p
        if p isa AbstractArray
            sz = size(p)
            l = prod(sz)
            pflat = reshape(view(psflat, (offset + 1):(offset + l)), sz)
            offset += l
            pflat
        else
            p
        end
    end
end


din = 2
dout = 3
dh = 128
nl = 8
x = rand(Float32, din, 128)

m = Chain(Dense(din, dh), [Dense(dh,dh) for _=1:nl-2]..., Dense(dh, dout))
fm = flatmodel(m)
p = Flux.params(m)

@btime $m($x)
@btime $fm($x)

nothing
#@btime gradient(() -> sum($m($x)), $p)
#@btime gradient((_m) -> sum(_m($x)), $m)


