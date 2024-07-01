import torch

def gen_random_multi_gaussian(mu, sample_num, sigma_scale=1.0):
    # mu: d dimension tensor
    # sample_num: int
    # sigma_scale: float
    # return: sample_num * d

    multi_gaussian_dim = mu.shape[0]
    # sigma_scale = 10.0
    # generate diagonal matrix
    sigma = torch.eye(multi_gaussian_dim)

    # change sigma scale
    sigma *= sigma_scale

    sampler = torch.distributions.MultivariateNormal(
        loc=mu, covariance_matrix=sigma
    )

    # sample_num_torch_size = torch.Size([sample_num])
    # samples: torch.Tensor = sampler.sample(sample_num_torch_size)
    samples: torch.Tensor = sampler.sample((sample_num, ))

    return samples


def gen_sanitiy_check_data():
    sample_num = 1000
    positive_num = sample_num // 2
    negative_num = sample_num - positive_num

    # positive_mu = torch.tensor([-100, -100, -100], dtype=torch.float32)
    positive_mu = torch.tensor([0, 0, 0], dtype=torch.float32)

    # negative_mu = torch.tensor([100, 100, 100], dtype=torch.float32)
    negative_mu = torch.tensor([1000, 1000, 1000], dtype=torch.float32)

    positive_samples = gen_random_multi_gaussian(positive_mu, positive_num)
    negative_samples = gen_random_multi_gaussian(negative_mu, negative_num)

    return positive_samples, negative_samples


