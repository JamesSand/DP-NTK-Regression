import torch

def gen_random_multi_gaussian(mu, sample_num, sigma_scale=1.0):
    # mu = torch.FloatTensor([1, 2, 0])
    multi_gaussian_dim = mu.shape[0]
    # sigma_scale = 10.0
    # generate diagonal matrix
    sigma = torch.eye(multi_gaussian_dim)

    # change sigma scale
    sigma *= sigma_scale

    sampler = torch.distributions.MultivariateNormal(
        loc=mu, covariance_matrix=sigma
    )
    samples: torch.Tensor = sampler.sample((sample_num, ))
    # samples.shape [100000, 3]

    return samples

    # print(samples.shape)

    # # estimation of mu and 
    # new_mu = samples.mean(dim=0)
    # new_sigma = (samples - mu).T @ (samples - mu) / len(samples)
    
    # print(new_mu.round())
    # print(new_sigma.round())

    # breakpoint()
    # print()


