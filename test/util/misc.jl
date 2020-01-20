@testset "whiten/whiten!" begin
    x = rand(1000)
    x[1:100] .*= 30
    LyceumAI.whiten!(x)
    @test std(x) ≈ 1
    @test abs(mean(x)) < sqrt(eps(eltype(x)))
end

@testset "TimeFeatures" begin
    dimobs = 3
    N = 100
    let o = [1, 2, 3], c = [2, 4, 6], dt = 0.01
        A = rand(dimobs, N)
        op = LyceumAI.TimeFeatures(o, c, dt)
        B = op(A)
        x = (0:(N - 1)) .* dt
        @test B[1:dimobs, :] == A
        @test all(Iterators.product(1:dimobs, 1:N)) do (k, j)
            i = k + dimobs
            B[i,j] == c[k] * x[j]^o[k]
        end
    end
    let o = [1, 2, 3], c = [2, 4, 6], dt = 0.01
        op = LyceumAI.TimeFeatures([1,2,3], [1,1,1], 0.01)
        A1 = [rand(3, 5), rand(3, 4)]
        A2 = hcat(A1...)
        B = op(A1)
        @test op(A2[:, 1:5]) == B[:, 1:5]
        @test op(A2[:, 6:end]) == B[:, 6:end]
    end
    let o = [1, 2, 3], c = [2, 4, 6], dt = 0.01
        op = LyceumAI.TimeFeatures([1,2,3], [1,1,1], 0.01)
        A1 = rand(3, 2)
        B1 = op(A1, [5, 10])
        A2 = [rand(3, 5), rand(3, 10)]
        B2 = op(A2)
        @test B1[(dimobs + 1):end, 1] == B2[(dimobs + 1):end, 5]
        @test B1[(dimobs + 1):end, 2] == B2[(dimobs + 1):end, end]
    end
end