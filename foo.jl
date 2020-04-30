using Distributions
using Statistics
using UnicodePlots
using LinearAlgebra
using Random
using Random: rand!

nu, T, K = 2, 10000, 8
covar = [0.1 0; 0 0.5]

println("distributions")
d = MvNormal([0,0], covar)
noise1 = zeros(nu, T, K)
rand!(d, reshape(noise1, Val(2)))
display(lineplot(noise1[1,:,1]))
display(lineplot(noise1[2,:,1]))

println("manual")
noise2 = randn(nu, T, K)
covar_ul = convert(AbstractMatrix, cholesky(covar).UL)
lmul!(covar_ul, reshape(noise2, Val(2)))
display(lineplot(noise2[1,:,1]))
display(lineplot(noise2[2,:,1]))

@info "" cov(noise1[1,:,1]) cov(noise2[1, :, 1]) # should be ~ 0.1
@info "" cov(noise1[2,:,1]) cov(noise2[2, :, 1]) # should be ~ 0.5


function combinetrajectories!(costs, meantrajectory, noise)
    beta = minimum(costs)
    eta = zero(Float64)
    for k = 1:size(noise, 3)
        @inbounds costs[k] = softcost = exp((beta - costs[k]) / 0.1)
        eta += softcost
    end

    costs ./= eta

    for k = 1:size(noise, 3), t = 1:size(noise, 2), u = 1:size(noise, 1)
        meantrajectory[u, t] += costs[k] * noise[u, t, k]
    end

end

meantrajectory = zeros(2, size(noise1, 2))
costs = zeros(size(noise1, 3))
costs .= 1/length(costs)

for i=1:5
    combinetrajectories!(costs, meantrajectory, noise1)
    @info extrema(meantrajectory[1,:])
    @info extrema(meantrajectory[2,:])
end
