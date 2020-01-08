@testset "whiten/whiten!" begin
    x = rand(1000)
    x[1:100] .*= 30
    LyceumAI.whiten!(x)
    @test std(x) ≈ 1
    @test abs(mean(x)) < sqrt(eps(eltype(x)))
end