module TestVectorProducts

@testset "FVP" begin
    # test for correct output and no side effects
    d1, d2 = 5, 10
    @test let glls = rand(d1, d2), x = rand(d1)
        glls2 = copy(glls)
        x2 = copy(x)
        y1 = LyceumAI.FVP(glls2) * x2
        y2 = glls * transpose(glls) * x
        isapprox(y1, y2) && eltype(y1) === eltype(glls) && x == x2 && glls == glls2
    end
    @test let glls = rand(d1, d2), x = rand(Float32, d1)
        glls2 = copy(glls)
        x2 = copy(x)
        y1 = LyceumAI.FVP{Float32}(glls2) * x2
        y2 = glls * transpose(glls) * x
        isapprox(y1, y2) && eltype(y1) === Float32 && x == x2 && glls == glls2
    end
    @test let glls = rand(d1, d2), x = rand(d1), y1 = zero(x)
        glls2 = copy(glls)
        x2 = copy(x)
        mul!(y1, LyceumAI.FVP(glls2), x2)
        y2 = glls * transpose(glls) * x
        isapprox(y1, y2) && x == x2 && glls == glls2
    end
    @test let glls = rand(d1, d2), x = rand(d1), y1 = zero(x)
        glls2 = copy(glls)
        x2 = copy(x)
        mul!(y1, LyceumAI.FVP(glls2, true), x2)
        y2 = 1/d2 * glls * transpose(glls) * x
        isapprox(y1, y2) && x == x2 && glls == glls2
    end
end

end
