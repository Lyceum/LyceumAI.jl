using MacroTools
using MacroTools: splitdef, combinedef

macro mustimplement(sig)
    @info sig
    def = Dict{Symbol,Any}()
    #if @capture(sig, (::T_)(xs__))
    #    @info "T" T xs
    #    def[:name]
    if @capture(sig, f_(xs__))
        @info "F" f xs
    else
        error()
    end
    :($(esc(sig)) = error("must implement ", $(string(sig))))
end

abstract type Dam end
struct Bar <: Dam end
struct Baz end

@mustimplement (::Dam)(x)
@mustimplement foo(x)

