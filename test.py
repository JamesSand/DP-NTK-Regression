# from run_10_cls import cal_k
import math


def cal_k(eps, delta, beta):
    eta = 7e-3
    n = 1e3
    k_bound = (eps * eps * eta * eta) / (8 * math.log(1 / delta) * n * n * beta * beta)
    k = int(math.floor(k_bound))
    return k

# fix beta and delta
beta = 1e-6
delta = 1e-3 

# we run different eps exponent, from (0.5, 1.5)
eps_exponent_list = [0.5 + i * 0.1 for i in range(11)]

for eps_exponent in eps_exponent_list:
    eps = 10 ** eps_exponent 

    # calculate number of Gaussian Samples according to eps
    k = cal_k(eps, delta, beta)

    print(eps, k)





