import torch

mu = torch.FloatTensor([1, 2, 0])
multi_gaussian_dim = mu.shape[0]
sigma_scale = 10.0
sigma = torch.eye(multi_gaussian_dim)
sigma *= sigma_scale
print(sigma)
# sigma = torch.FloatTensor([
#     [2, 0, 0],
#     [0, 5, 0],
#     [0, 0, 1]
# ])

sampler = torch.distributions.MultivariateNormal(
    loc=mu, covariance_matrix=sigma
)
samples: torch.Tensor = sampler.sample((100000, ))
# samples.shape [100000, 3]

print(samples.shape)

new_mu = samples.mean(dim=0)
new_sigma = (samples - mu).T @ (samples - mu) / len(samples)
 
print(new_mu.round())
print(new_sigma.round())

breakpoint()
print()


