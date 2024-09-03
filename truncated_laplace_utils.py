import numpy as np
import torch
from scipy.stats import rv_continuous
import math
import time
from torch.distributions.laplace import Laplace

class truncated_laplace_gen(rv_continuous):
    def __init__(self, loc=0, scale=1, lower=-np.inf, upper=np.inf):
        super().__init__(a=lower, b=upper)
        # self.x = x
        # self.p = p
        self.loc = loc, 
        self.scale = scale
        self.lower = lower
        self.upper = upper

        lower_cdf = 0.5 * np.exp((self.lower - self.loc[0]) / self.scale)
        upper_cdf = 1 - 0.5 * np.exp(-(self.upper - self.loc[0]) / self.scale)
        
        self.normalization_constant = upper_cdf - lower_cdf

    def _pdf(self, x):

        # no probability for generating value outside the bound
        if x <= self.lower or x >= self.upper:
            return 0.0
        
        laplace_pdf = 1/(2*self.scale) * np.exp(-np.abs(x - self.loc[0])/self.scale)
        return laplace_pdf / self.normalization_constant
    

def add_truncated_laplace_noise(beta, eps, delta, x_data):
    # x_data: n * 512
    n, d = x_data.shape
    sensitivity = math.sqrt(d) * beta
    truncated_bound = (sensitivity / eps) * math.log(1 + (math.exp(eps) - 1) / (2 * delta))
    laplace_scale = sensitivity / eps
    loc = 0.0

    truncated_laplace = truncated_laplace_gen(loc=loc, scale=laplace_scale, lower=-truncated_bound, upper=truncated_bound)

    truncated_laplace_noise = truncated_laplace.rvs(size=n * d)
    truncated_laplace_noise_ts = torch.from_numpy(truncated_laplace_noise)
    truncated_laplace_noise_ts = truncated_laplace_noise_ts.reshape(n, d)

    return x_data + truncated_laplace_noise_ts

def add_laplace_noise(beta, eps, x_data):
    n, d = x_data.shape
    sensitivity = math.sqrt(d) * beta
    lapalce_scale = sensitivity / eps

    laplace_gen = Laplace(0.0, lapalce_scale)
    laplace_noise = laplace_gen.sample([n, d])

    return x_data + laplace_noise

    

def test_truncated_bound(beta, eps, delta):
    # x_data: n * 512
    # n, d = x_data.shape
    d = 512
    sensitivity = math.sqrt(d) * beta
    truncated_bound = (sensitivity / eps) * math.log(1 + (math.exp(eps) - 1) / (2 * delta))

    return truncated_bound



if __name__ == "__main__":

    # # fix beta and delta
    # beta = 1e-6
    # delta = 1e-3

    # # we run different eps exponent, from (0.5, 1.5)
    # eps_exponent_list = [0.5 + i * 0.1 for i in range(11)]
    # for eps_exponent in eps_exponent_list:
    #     eps = 10 ** eps_exponent
    #     truncated_bound = test_truncated_bound(beta, eps, delta)
    #     print(eps_exponent, truncated_bound)

    # exit(0)

    start_time = time.time()

    laplace_gen = Laplace(0.0, 1.0)
    laplace_noise = laplace_gen.sample([10000, 512])

    end_time = time.time()

    print(f"elapse time {end_time - start_time}")

    breakpoint()


    start_time = time.time()

    # the following is test for above class
    # Define parameters
    loc = 0       # Mean of the Laplace distribution
    scale = 1     # Scale parameter (similar to standard deviation)
    lower = -2    # Lower truncation bound
    upper = 2     # Upper truncation bound

    # Create an instance of the distribution without shape defaults
    truncated_laplace = truncated_laplace_gen(loc=loc, scale=scale, lower=lower, upper=upper)

    # Generate random samples
    samples = truncated_laplace.rvs(size=10000)
    samples_ts = torch.from_numpy(samples)
    samples_ts = samples_ts.reshape(10, 1000)
    print(samples_ts.shape)

    end_time = time.time()

    print(f"elapse time {end_time - start_time}")

    breakpoint()
    # print(type(samples))
    # breakpoint()

    # Print some samples
    print(samples[:10])



