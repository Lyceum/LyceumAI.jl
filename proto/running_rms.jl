#class RunningMeanStd(object):
#    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
#    def __init__(self, epsilon=1e-4, shape=()):
#        self.mean = np.zeros(shape, 'float64')
#        self.var = np.ones(shape, 'float64')
#        self.count = epsilon
#
#    def update(self, x):
#        batch_mean = np.mean(x, axis=0)
#        batch_var = np.var(x, axis=0)
#        batch_count = x.shape[0]
#        self.update_from_moments(batch_mean, batch_var, batch_count)
#
#    def update_from_moments(self, batch_mean, batch_var, batch_count):
#        self.mean, self.var, self.count = update_mean_var_count_from_moments(
#            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)
#
#def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
#    delta = batch_mean - mean
#    tot_count = count + batch_count
#
#    new_mean = mean + delta * batch_count / tot_count
#    m_a = var * count
#    m_b = batch_var * batch_count
#    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
#    new_var = M2 / tot_count
#    new_count = tot_count
#
#    jreturn new_mean, new_var, new_count

using Parameters

import Statistics: mean, std, var

# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
mutable struct WelfordRMS{T}
    mean::T
    M2::T
    count::Int
end

WelfordRMS(dtype::Type{<:Real}) = WelfordRMS{dtype}(0, 0, 0)

function fit!(m::WelfordRMS, xs)
    @unpack mean, M2, count = m
    for x in xs
        count += 1
        delta1 = x - mean
        mean += delta1 / count
        delta2 = x - mean
        M2 += delta1 * delta2
    end
    @pack! m = mean, M2, count
    m
end

mean(m::WelfordRMS) = m.mean

var(m::WelfordRMS; corrected::Bool = true) =
    ifelse(corrected, m.M2 / m.count, m.M2 / (m.count - 1))

std(m::WelfordRMS; corrected::Bool = true) = sqrt(var(m, corrected = corrected))
